import { useEffect, useRef, useState } from "react";

export default function useAgentWorker() {
  const workerRef = useRef(null);
  const runStartedAtRef = useRef(null);
  const [modelPhase, setModelPhase] = useState("idle");
  const [runPhase, setRunPhase] = useState("idle");
  const [progress, setProgress] = useState(null);
  const [trace, setTrace] = useState([]);
  const [error, setError] = useState("");
  const [stats, setStats] = useState(null);
  const [plan, setPlan] = useState(null);
  const [interaction, setInteraction] = useState(null);
  const [artifact, setArtifact] = useState(null);
  const [elapsedMs, setElapsedMs] = useState(0);

  useEffect(() => {
    const worker = new Worker(
      new URL("../workers/agent.worker.js", import.meta.url),
      {
        type: "module",
      }
    );
    workerRef.current = worker;

    worker.addEventListener(
      "message",
      (event) => {
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
            });
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
                });
              },
              (locationError) => {
                worker.postMessage({
                  type: "location_response",
                  id: message.id,
                  error: locationError.message,
                });
              },
              {
                enableHighAccuracy: false,
                maximumAge: 300000,
                timeout: 10000,
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
          setError(message.message || "Unknown error");
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

  const send = (request) =>
    workerRef.current?.postMessage(request);

  const load = () => {
    setError("");
    setModelPhase("loading");
    send({ type: "load" });
  };

  const generate = (prompt, allowedTools, documentSegments = []) => {
    runStartedAtRef.current = performance.now();
    setElapsedMs(0);
    setError("");
    setTrace([]);
    setStats(null);
    setPlan(null);
    setInteraction(null);
    setArtifact(null);
    setRunPhase("thinking");
    send({ type: "generate", prompt, allowedTools: [...allowedTools], documentSegments });
  };

  const submitInteraction = (answer) => {
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
