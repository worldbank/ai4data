import { useEffect, useRef, useState } from "react";
import type {
  ActionPlan,
  BrowserArtifact,
  LoadProgress,
  ModelPhase,
  RunStats,
  RunPhase,
  TraceEvent,
  UserInteraction,
  WorkerRequest,
  WorkerResponse,
} from "../types";

export default function useAgentWorker() {
  const workerRef = useRef<Worker | null>(null);
  const runStartedAtRef = useRef<number | null>(null);
  const [modelPhase, setModelPhase] = useState<ModelPhase>("idle");
  const [runPhase, setRunPhase] = useState<RunPhase>("idle");
  const [progress, setProgress] = useState<LoadProgress | null>(null);
  const [trace, setTrace] = useState<TraceEvent[]>([]);
  const [error, setError] = useState("");
  const [stats, setStats] = useState<RunStats | null>(null);
  const [plan, setPlan] = useState<ActionPlan | null>(null);
  const [interaction, setInteraction] = useState<UserInteraction | null>(null);
  const [artifact, setArtifact] = useState<BrowserArtifact | null>(null);
  const [elapsedMs, setElapsedMs] = useState(0);

  useEffect(() => {
    const worker = new Worker(
      new URL("../workers/agent.worker.ts", import.meta.url),
      {
        type: "module",
      }
    );
    workerRef.current = worker;

    worker.addEventListener(
      "message",
      (event: MessageEvent<WorkerResponse>) => {
        const message = event.data;

        if (message.type === "loading") {
          setModelPhase("loading");
          setProgress(message.data);
        } else if (message.type === "ready") {
          setModelPhase("ready");
        } else if (message.type === "start") {
          setRunPhase("thinking");
        } else if (message.type === "turn") {
          setTrace((current) => {
            const existingIndex = current.findIndex(
              (item) => item.id === message.data.id
            );
            if (existingIndex === -1) return [...current, message.data];
            return current.map((item, index) =>
              index === existingIndex ? message.data : item
            );
          });
        } else if (message.type === "tool") {
          setTrace((current) => {
            const existingIndex = current.findIndex(
              (item) => item.id === message.data.id
            );
            if (existingIndex === -1) return [...current, message.data];
            return current.map((item, index) =>
              index === existingIndex ? message.data : item
            );
          });
        } else if (message.type === "plan") {
          setPlan(message.data);
        } else if (message.type === "interaction") {
          setInteraction(message.data);
          setRunPhase("waiting");
        } else if (message.type === "location_request") {
          if (!("geolocation" in navigator)) {
            worker.postMessage({
              type: "location_response",
              id: message.id,
              error: "Geolocation is not supported by this browser.",
            } satisfies WorkerRequest);
          } else {
            navigator.geolocation.getCurrentPosition(
              (position) => {
                worker.postMessage({
                  type: "location_response",
                  id: message.id,
                  location: {
                    latitude: position.coords.latitude,
                    longitude: position.coords.longitude,
                    accuracy: position.coords.accuracy,
                  },
                } satisfies WorkerRequest);
              },
              (locationError) => {
                worker.postMessage({
                  type: "location_response",
                  id: message.id,
                  error: locationError.message,
                } satisfies WorkerRequest);
              },
              {
                enableHighAccuracy: false,
                maximumAge: 300_000,
                timeout: 10_000,
              }
            );
          }
        } else if (message.type === "artifact") {
          setArtifact(message.data);
        } else if (message.type === "metrics") {
          setStats(message.data);
        } else if (message.type === "complete") {
          setRunPhase("complete");
          setStats(message.data);
          setElapsedMs(
            runStartedAtRef.current === null
              ? message.data.elapsedMs
              : performance.now() - runStartedAtRef.current
          );
        } else {
          setModelPhase((current) =>
            current === "loading" ? "error" : current
          );
          setRunPhase("error");
          if (runStartedAtRef.current !== null) {
            setElapsedMs(performance.now() - runStartedAtRef.current);
          }
          setError(message.message);
        }
      }
    );

    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (
      (runPhase !== "thinking" && runPhase !== "waiting") ||
      runStartedAtRef.current === null
    ) {
      return;
    }
    const updateElapsed = () => {
      if (runStartedAtRef.current !== null) {
        setElapsedMs(performance.now() - runStartedAtRef.current);
      }
    };
    updateElapsed();
    const interval = window.setInterval(updateElapsed, 500);
    return () => window.clearInterval(interval);
  }, [runPhase]);

  const send = (request: WorkerRequest) =>
    workerRef.current?.postMessage(request);

  const load = () => {
    setError("");
    setModelPhase("loading");
    send({ type: "load" });
  };

  const generate = (prompt: string, allowedTools: readonly string[]) => {
    runStartedAtRef.current = performance.now();
    setElapsedMs(0);
    setError("");
    setTrace([]);
    setStats(null);
    setPlan(null);
    setInteraction(null);
    setArtifact(null);
    setRunPhase("thinking");
    send({ type: "generate", prompt, allowedTools: [...allowedTools] });
  };

  const submitInteraction = (answer: string) => {
    if (!interaction || !answer.trim()) return;
    send({
      type: "interaction_response",
      id: interaction.id,
      answer: answer.trim(),
    });
    setInteraction(null);
    setRunPhase("thinking");
  };

  const stop = () => {
    send({ type: "stop" });
    setInteraction(null);
  };

  return {
    artifact,
    elapsedMs,
    error,
    generate,
    interaction,
    load,
    modelPhase,
    plan,
    progress,
    runPhase,
    stats,
    stop,
    submitInteraction,
    trace,
  };
}
