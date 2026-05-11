import { useEffect, useRef, useState } from "react";
import type { FrameMessage, MetaMessage } from "./types";

export type DecodedFrame = {
  msg: FrameMessage;
  jpegByCam: Map<number, Blob>; // cam_id -> JPEG blob
};

function parseBinaryFrame(buf: ArrayBuffer): DecodedFrame | null {
  const view = new DataView(buf);
  const headerLen = view.getUint32(0, false);
  const headerJson = new TextDecoder().decode(new Uint8Array(buf, 4, headerLen));
  const msg = JSON.parse(headerJson) as FrameMessage;
  const blobBase = 4 + headerLen;
  const jpegByCam = new Map<number, Blob>();
  for (const b of msg.blobs) {
    const slice = buf.slice(blobBase + b.offset, blobBase + b.offset + b.length);
    jpegByCam.set(b.cam_id, new Blob([slice], { type: "image/jpeg" }));
  }
  return { msg, jpegByCam };
}

export type WebSocketState = {
  frame: DecodedFrame | null;
  prevFrame: DecodedFrame | null;  // the frame received just before `frame`
  receiveTime: number;             // performance.now() when `frame` arrived
  meta: MetaMessage | null;
  status: "idle" | "connecting" | "open" | "closed";
};

export function useWebSocket(enabled: boolean): WebSocketState {
  const [frame, setFrame] = useState<DecodedFrame | null>(null);
  const [prevFrame, setPrevFrame] = useState<DecodedFrame | null>(null);
  const [receiveTime, setReceiveTime] = useState<number>(0);
  const [meta, setMeta] = useState<MetaMessage | null>(null);
  const [status, setStatus] = useState<"idle" | "connecting" | "open" | "closed">("idle");
  const wsRef = useRef<WebSocket | null>(null);

  // Keep a ref of the current frame so the closure in onmessage always sees it
  const currentFrameRef = useRef<DecodedFrame | null>(null);

  useEffect(() => {
    if (!enabled) {
      wsRef.current?.close();
      wsRef.current = null;
      setStatus("idle");
      setMeta(null);
      currentFrameRef.current = null;
      return;
    }
    setStatus("connecting");
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${window.location.host}/ws`;
    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;

    ws.onopen = () => setStatus("open");
    ws.onclose = () => setStatus("closed");
    ws.onerror = () => setStatus("closed");
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        // JSON text frames: meta message
        try {
          const parsed = JSON.parse(ev.data);
          if (parsed.type === "meta") {
            setMeta(parsed as MetaMessage);
          }
        } catch {
          // ignore malformed text
        }
        return;
      }
      const decoded = parseBinaryFrame(ev.data as ArrayBuffer);
      if (decoded) {
        const now = performance.now();
        setPrevFrame(currentFrameRef.current);
        currentFrameRef.current = decoded;
        setFrame(decoded);
        setReceiveTime(now);
      }
    };
    return () => {
      ws.close();
      wsRef.current = null;
      currentFrameRef.current = null;
    };
  }, [enabled]);

  return { frame, prevFrame, receiveTime, meta, status };
}

// Tiny helper to convert a JPEG Blob to an HTMLImageElement once
export function blobToImage(blob: Blob): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(blob);
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("blob decode failed"));
    };
    img.src = url;
  });
}
