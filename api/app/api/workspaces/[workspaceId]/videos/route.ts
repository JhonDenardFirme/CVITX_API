import { NextResponse } from "next/server";
import { apiFetch } from "../../../_proxy/client";

export async function GET(_: Request, { params }: { params: { workspaceId: string } }) {
  const { workspaceId } = params;
  const data = await apiFetch(`/workspaces/${workspaceId}/videos`);
  return NextResponse.json(data);
}

export async function POST(req: Request, { params }: { params: { workspaceId: string } }) {
  const { workspaceId } = params;
  const body = await req.json();
  const data = await apiFetch(`/workspaces/${workspaceId}/videos`, {
    method: "POST",
    body: JSON.stringify(body),
    headers: { "Content-Type": "application/json" },
  });
  return NextResponse.json(data);
}
