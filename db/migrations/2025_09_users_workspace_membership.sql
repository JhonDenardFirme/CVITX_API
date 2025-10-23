-- === Extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS citext;

-- === Enums
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname='user_role_enum') THEN
    CREATE TYPE user_role_enum AS ENUM ('admin','user');
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname='account_request_status_enum') THEN
    CREATE TYPE account_request_status_enum AS ENUM
      ('pending','approved','rejected','fulfilled','cancelled');
  END IF;
END $$;

-- === updated_at helper
CREATE OR REPLACE FUNCTION set_updated_at() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN NEW.updated_at := now(); RETURN NEW; END $$;

-- === Users
CREATE TABLE IF NOT EXISTS public.users (
  id                     uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  email                  citext      NOT NULL UNIQUE,
  password_hash          text        NOT NULL,
  first_name             text,
  last_name              text,
  affiliation_name       text,
  avatar_s3_key          text,
  role                   user_role_enum  NOT NULL DEFAULT 'user',
  is_active              boolean     NOT NULL DEFAULT true,
  force_password_reset   boolean     NOT NULL DEFAULT true,
  email_verified_at      timestamptz,
  last_login_at          timestamptz,
  password_updated_at    timestamptz,
  created_at             timestamptz NOT NULL DEFAULT now(),
  updated_at             timestamptz NOT NULL DEFAULT now(),
  deleted_at             timestamptz
);
DROP TRIGGER IF EXISTS trg_users_set_updated_at ON public.users;
CREATE TRIGGER trg_users_set_updated_at
BEFORE UPDATE ON public.users
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- === Account Requests
CREATE TABLE IF NOT EXISTS public.account_requests (
  id                 uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  email              citext      NOT NULL,
  first_name         text,
  middle_initial     text,
  last_name          text,
  affiliation_name   text,
  affiliation_role   text,
  purpose_of_use     text,
  status             account_request_status_enum NOT NULL DEFAULT 'pending',
  decided_by_user_id uuid REFERENCES public.users(id) ON DELETE SET NULL,
  decided_at         timestamptz,
  linked_user_id     uuid REFERENCES public.users(id) ON DELETE SET NULL,
  created_at         timestamptz NOT NULL DEFAULT now(),
  updated_at         timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_account_requests_pending_email
  ON public.account_requests (email)
  WHERE status = 'pending';
DROP TRIGGER IF EXISTS trg_account_requests_set_updated_at ON public.account_requests;
CREATE TRIGGER trg_account_requests_set_updated_at
BEFORE UPDATE ON public.account_requests
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- === Password reset / invite tokens
CREATE TABLE IF NOT EXISTS public.password_reset_tokens (
  id          uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     uuid        NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  token_hash  text        NOT NULL UNIQUE,
  purpose     text        NOT NULL DEFAULT 'reset',
  expires_at  timestamptz NOT NULL,
  used_at     timestamptz,
  created_at  timestamptz NOT NULL DEFAULT now()
);

-- === Membership (users ↔ workspaces)
CREATE TABLE IF NOT EXISTS public.workspace_members (
  workspace_id uuid NOT NULL REFERENCES public.workspaces(id) ON DELETE CASCADE,
  user_id      uuid NOT NULL REFERENCES public.users(id)      ON DELETE CASCADE,
  joined_at    timestamptz NOT NULL DEFAULT now(),
  invited_by_user_id uuid REFERENCES public.users(id) ON DELETE SET NULL,
  PRIMARY KEY (workspace_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_workspace_members_user ON public.workspace_members(user_id);

-- === Extend workspaces safely
ALTER TABLE public.workspaces
  ADD COLUMN IF NOT EXISTS owner_user_id uuid REFERENCES public.users(id) ON DELETE SET NULL,
  ADD COLUMN IF NOT EXISTS created_at    timestamptz NOT NULL DEFAULT now(),
  ADD COLUMN IF NOT EXISTS updated_at    timestamptz NOT NULL DEFAULT now(),
  ADD COLUMN IF NOT EXISTS deleted_at    timestamptz;

DROP TRIGGER IF EXISTS trg_workspaces_set_updated_at ON public.workspaces;
CREATE TRIGGER trg_workspaces_set_updated_at
BEFORE UPDATE ON public.workspaces
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- === Global sequence & generator for CTX#### codes
CREATE SEQUENCE IF NOT EXISTS public.workspace_code_seq START WITH 1001;

CREATE OR REPLACE FUNCTION public.gen_workspace_code() RETURNS text
LANGUAGE plpgsql AS $$
DECLARE n bigint;
BEGIN
  n := nextval('public.workspace_code_seq');
  RETURN 'CTX' || n::text;
END $$;

-- Set default for workspace_code if none defined yet
DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_attrdef d
    JOIN pg_class c ON c.oid = d.adrelid
    JOIN pg_attribute a ON a.attrelid=d.adrelid AND a.attnum=d.adnum
    WHERE c.relname='workspaces' AND a.attname='workspace_code'
  ) THEN
    ALTER TABLE public.workspaces
      ALTER COLUMN workspace_code SET DEFAULT public.gen_workspace_code();
  END IF;
END $$;

-- === Enforce "max 3 active workspaces per owner"
CREATE OR REPLACE FUNCTION public.enforce_workspace_limit() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE cnt int;
BEGIN
  IF NEW.owner_user_id IS NULL THEN
    RETURN NEW;
  END IF;

  IF (TG_OP = 'INSERT')
     OR (TG_OP='UPDATE' AND (NEW.owner_user_id IS DISTINCT FROM OLD.owner_user_id
                             OR (OLD.deleted_at IS NOT NULL AND NEW.deleted_at IS NULL)
                             OR (OLD.deleted_at IS NULL  AND NEW.deleted_at IS NULL))) THEN
    SELECT COUNT(*) INTO cnt
    FROM public.workspaces
    WHERE owner_user_id = NEW.owner_user_id
      AND deleted_at IS NULL
      AND id <> COALESCE(NEW.id, '00000000-0000-0000-0000-000000000000')::uuid;

    IF cnt >= 3 AND NEW.deleted_at IS NULL THEN
      RAISE EXCEPTION 'Owner % already has 3 active workspaces', NEW.owner_user_id
        USING ERRCODE = '23514';
    END IF;
  END IF;

  RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS trg_workspaces_limit ON public.workspaces;
CREATE TRIGGER trg_workspaces_limit
BEFORE INSERT OR UPDATE ON public.workspaces
FOR EACH ROW EXECUTE FUNCTION public.enforce_workspace_limit();

-- === Ensure owner is a member automatically
CREATE OR REPLACE FUNCTION public.ensure_owner_membership() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.owner_user_id IS NOT NULL AND NEW.deleted_at IS NULL THEN
    INSERT INTO public.workspace_members(workspace_id, user_id)
    VALUES (NEW.id, NEW.owner_user_id)
    ON CONFLICT DO NOTHING;
  END IF;
  RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS trg_workspaces_owner_membership ON public.workspaces;
CREATE TRIGGER trg_workspaces_owner_membership
AFTER INSERT OR UPDATE ON public.workspaces
FOR EACH ROW EXECUTE FUNCTION public.ensure_owner_membership();
