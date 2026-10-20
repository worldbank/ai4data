import type {
  ActionPlan,
  BrowserLocation,
  PlanStepStatus,
  UserInteraction,
} from "./types";

export interface ToolExecutionContext {
  getActivePlan: () => ActionPlan | null;
  setActivePlan: (plan: ActionPlan) => void;
  askUser: (
    interaction: Omit<UserInteraction, "id">
  ) => Promise<Record<string, unknown>>;
  getLocation: () => Promise<BrowserLocation | { error: string }>;
  researchWikipedia: (
    question: string,
    language: string
  ) => Promise<Record<string, unknown>>;
}

export interface AgentToolDefinition {
  name: string;
  label: string;
  summary: string;
  description: string;
  returns: string;
  schema: Record<string, unknown>;
  execute: (
    arguments_: Record<string, unknown>,
    context: ToolExecutionContext
  ) => Promise<Record<string, unknown>>;
}

function toStringArray(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value
      .map(String)
      .map((item) => item.trim())
      .filter(Boolean);
  }
  if (typeof value !== "string") return [];
  const quoted = [...value.matchAll(/['"]([^'"]+)['"]/g)].map(
    (match) => match[1]
  );
  if (quoted.length > 0) return quoted;
  return value
    .replace(/^\[|\]$/g, "")
    .split(/;|\n/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function createActionPlan(
  arguments_: Record<string, unknown>,
  context: ToolExecutionContext
): Record<string, unknown> {
  const goal = String(arguments_.goal ?? "Complete the current mission");
  const requestedSteps = toStringArray(arguments_.steps).slice(0, 5);
  const steps =
    requestedSteps.length >= 1
      ? requestedSteps
      : [
          "Clarify the research scope",
          "Gather reliable evidence",
          "Synthesize the cited research answer",
        ];
  const requestedPriority = String(arguments_.priority ?? "medium");
  const priority = ["low", "medium", "high"].includes(requestedPriority)
    ? (requestedPriority as ActionPlan["priority"])
    : "medium";
  const plan: ActionPlan = {
    id: `local-plan-${crypto.randomUUID().slice(0, 8)}`,
    goal,
    priority,
    status: "active",
    currentStep: 1,
    steps: steps.map((title, index) => ({
      id: `step-${index + 1}`,
      title,
      status: index === 0 ? "in_progress" : "pending",
    })),
    updatedAt: Date.now(),
  };
  context.setActivePlan(plan);
  return { stored_in: "conversation memory", plan };
}

function updateActionPlan(
  arguments_: Record<string, unknown>,
  context: ToolExecutionContext
): Record<string, unknown> {
  const activePlan = context.getActivePlan();
  if (!activePlan) {
    return { error: "No active plan. Call create_action_plan first." };
  }
  const index = Math.max(0, Number(arguments_.step_index ?? 1) - 1);
  if (!activePlan.steps[index]) {
    return { error: `Step ${index + 1} does not exist`, plan: activePlan };
  }
  const validStatuses: PlanStepStatus[] = [
    "pending",
    "in_progress",
    "completed",
    "blocked",
  ];
  const requestedStatus = String(arguments_.status ?? "in_progress");
  const status = validStatuses.includes(requestedStatus as PlanStepStatus)
    ? (requestedStatus as PlanStepStatus)
    : "in_progress";
  const getStepStatus = (
    stepStatus: PlanStepStatus,
    stepIndex: number
  ): PlanStepStatus => {
    if (status !== "in_progress") {
      return stepIndex === index ? status : stepStatus;
    }
    if (stepIndex < index) return "completed";
    if (stepIndex === index) return "in_progress";
    return stepStatus === "in_progress" ? "pending" : stepStatus;
  };
  const steps = activePlan.steps.map((step, stepIndex) => ({
    ...step,
    status: getStepStatus(step.status, stepIndex),
    ...(stepIndex === index && arguments_.note
      ? { note: String(arguments_.note) }
      : {}),
  }));

  if (status === "completed") {
    steps.forEach((step) => {
      if (step.status === "in_progress") step.status = "pending";
    });
    const nextIndex = steps.findIndex((step) => step.status === "pending");
    if (nextIndex >= 0) steps[nextIndex].status = "in_progress";
  }
  const currentIndex = steps.findIndex((step) => step.status === "in_progress");
  const allComplete = steps.every((step) => step.status === "completed");
  const plan: ActionPlan = {
    ...activePlan,
    steps,
    currentStep: currentIndex >= 0 ? currentIndex + 1 : index + 1,
    status: allComplete
      ? "completed"
      : status === "blocked"
        ? "blocked"
        : "active",
    updatedAt: Date.now(),
  };
  context.setActivePlan(plan);
  return { stored_in: "conversation memory", plan };
}

export const AGENT_TOOLS = [
  {
    name: "create_action_plan",
    label: "Create action plan",
    summary: "Create the ordered agenda that controls the mission workflow.",
    description:
      "Creates one to five execution steps based on the request and starts the first step before any other tool can run.",
    returns: "The session plan, current step, priority, and plan ID.",
    schema: {
      type: "function",
      function: {
        name: "create_action_plan",
        description:
          "Create the source-of-truth execution agenda before doing any other work. Steps must be agent actions that can finish in this conversation.",
        parameters: {
          type: "object",
          properties: {
            goal: { type: "string", description: "Mission goal" },
            steps: {
              type: "array",
              items: { type: "string" },
              minItems: 1,
              maxItems: 5,
            },
            priority: { type: "string", enum: ["low", "medium", "high"] },
          },
          required: ["goal", "steps", "priority"],
        },
      },
    },
    execute: async (
      arguments_: Record<string, unknown>,
      context: ToolExecutionContext
    ) => createActionPlan(arguments_, context),
  },
  {
    name: "update_action_plan",
    label: "Update action plan",
    summary: "Advance the current step as evidence arrives.",
    description:
      "Records evidence and step status. Starting a later step automatically completes all previous steps.",
    returns: "The complete updated plan and current workflow position.",
    schema: {
      type: "function",
      function: {
        name: "update_action_plan",
        description:
          "Update the action plan after starting or completing a workflow step.",
        parameters: {
          type: "object",
          properties: {
            step_index: {
              type: "integer",
              description: "One-based step index",
            },
            status: {
              type: "string",
              enum: ["pending", "in_progress", "completed", "blocked"],
            },
            note: { type: "string", description: "Evidence or progress note" },
          },
          required: ["step_index", "status", "note"],
        },
      },
    },
    execute: async (
      arguments_: Record<string, unknown>,
      context: ToolExecutionContext
    ) => updateActionPlan(arguments_, context),
  },
  {
    name: "ask_user",
    label: "Ask user",
    summary:
      "Narrow a broad request by asking about the user's preferred scope or perspective.",
    description:
      "Use after objective context is known when the request still permits meaningfully different answers. Ask about preferences such as time period, theme, audience, perspective, or depth. For a broad country-history question, for example, clarify whether to focus on an era or aspect rather than choosing one silently.",
    returns: "The user's selected or written answer.",
    schema: {
      type: "function",
      function: {
        name: "ask_user",
        description:
          "Ask one focused question to narrow the user's preferred scope, perspective, audience, time period, theme, or level of detail. Use this even after factual context such as location is known when the research question remains broad. Prefer select when a short set of meaningful choices exists.",
        parameters: {
          type: "object",
          properties: {
            question: { type: "string" },
            response_type: { type: "string", enum: ["text", "select"] },
            options: { type: "array", items: { type: "string" }, maxItems: 6 },
            placeholder: { type: "string" },
            allow_custom: { type: "boolean" },
          },
          required: ["question", "response_type"],
        },
      },
    },
    execute: async (
      arguments_: Record<string, unknown>,
      context: ToolExecutionContext
    ) => {
      const options = toStringArray(arguments_.options).slice(0, 6);
      const responseType =
        arguments_.response_type === "select" && options.length > 0
          ? "select"
          : "text";
      return context.askUser({
        question: String(arguments_.question ?? "What should I know?"),
        responseType,
        options: responseType === "select" ? options : [],
        placeholder: arguments_.placeholder
          ? String(arguments_.placeholder)
          : undefined,
        allowCustom: Boolean(arguments_.allow_custom),
      });
    },
  },
  {
    name: "search_wikipedia",
    label: "Wikipedia research agent",
    summary: "Delegate focused, cited research to an isolated local agent.",
    description:
      "Searches and reads a bounded set of Wikipedia pages, then uses an isolated model turn to return only distilled evidence and canonical sources.",
    returns:
      "A concise research brief, caveats, and source URLs without raw page text.",
    schema: {
      type: "function",
      function: {
        name: "search_wikipedia",
        description:
          "Delegate a focused research question to an isolated Wikipedia research agent. It searches and reads relevant pages, then returns a concise cited brief. Ask a complete question; raw articles are not added to your context.",
        parameters: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "The complete research question to investigate",
            },
            language: {
              type: "string",
              description: "Wikipedia language code",
            },
          },
          required: ["query"],
        },
      },
    },
    execute: async (
      arguments_: Record<string, unknown>,
      context: ToolExecutionContext
    ) => {
      const question = String(arguments_.query ?? "").trim();
      if (!question) return { error: "A research question is required." };
      const requestedLanguage = String(
        arguments_.language ?? "en"
      ).toLowerCase();
      const language = /^[a-z]{2,3}$/.test(requestedLanguage)
        ? requestedLanguage
        : "en";
      return context.researchWikipedia(question, language);
    },
  },
  {
    name: "get_current_context",
    label: "Get current context",
    summary:
      "Read the user's current device context, including permission-gated physical location.",
    description:
      "Uses browser APIs to read current date, time, timezone, language, connectivity, and device coordinates. With location permission, coordinates are reverse geocoded into country, region, city, and locality. For questions that depend on where the user currently is, call this tool with include_location=true instead of asking the user to select a location.",
    returns:
      "Current device context and, when permission is granted, coordinates plus resolved country, country code, region, city, and locality.",
    schema: {
      type: "function",
      function: {
        name: "get_current_context",
        description:
          "Get the user's current device context. When a request depends on the user's present physical location, set include_location=true to request and resolve the device location into country, region, city, and locality instead of asking the user where they are. Repeated calls return the same cached result for this run.",
        parameters: {
          type: "object",
          properties: {
            include_location: {
              type: "boolean",
              description:
                "Whether to resolve the user's current physical location. Defaults to true when omitted; set false only when date and time context is sufficient.",
            },
          },
          required: [],
        },
      },
    },
    execute: async (
      arguments_: Record<string, unknown>,
      context: ToolExecutionContext
    ) => {
      const now = new Date();
      const result: Record<string, unknown> = {
        iso_datetime: now.toISOString(),
        local_datetime: now.toLocaleString(),
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        language: navigator.language,
        online: navigator.onLine,
      };
      const includeLocation = arguments_.include_location !== false;
      if (includeLocation) result.location = await context.getLocation();
      return result;
    },
  },
] as const satisfies readonly AgentToolDefinition[];

export type ToolName = (typeof AGENT_TOOLS)[number]["name"];
