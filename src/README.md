# Monadic Context Engineering in TypeScript: A Practical Implementation

This document provides a practical, robust implementation of the concepts outlined in the research paper **"Monadic Context Engineering: A Principled Algebraic Framework for AI Agent Orchestration"** by Yifan Zhang. It uses TypeScript and the `fp-ts` library to build a type-safe, composable, and resilient framework for orchestrating AI agents.

> The goal is to move from brittle, imperative agent scripts to a principled, functional architecture where state management, error handling, and asynchronicity are handled intrinsically by the algebraic properties of the abstraction.

We will build the `AgentMonad` as described in the paper—a monad transformer stack of `StateT S (EitherT E IO) A`—and use it to create a simple but powerful research agent that explicitly calls the `@google/gemini-cli` tool.

---

## 1. Core Concepts

The `AgentMonad` is the core of this framework. It represents a computation that is:

1.  **Asynchronous & Side-Effecting (`Task`)**: It performs actions in the real world (like API calls) that take time.
2.  **Fallible (`Either`)**: Any step in the computation can fail, and the entire workflow will short-circuit gracefully.
3.  **Stateful (`State`)**: It implicitly carries an agent's state (memory, history, etc.) through the entire workflow, ensuring state integrity.

In `fp-ts`, this stack is represented by the type `StateT<S, TaskEither<E, A>, B>`, where `S` is the state type, `E` is the error type, and `B` is the value produced by the computation.

---

## 2. Project Setup

To run this project, you need Node.js and npm installed.

1.  **Initialize Project & Install Dependencies:**
    ```bash
    git clone https://github.com/yifanzhang-pro/monadic-context-engineering.git
    npm init -y
    npm install typescript ts-node fp-ts
    ```

2.  **Initialize TypeScript Configuration:**
    ```bash
    npx tsc --init
    ```
    This will create a `tsconfig.json` file with default settings, which is sufficient for this project.

3.  **Create Project Structure:**
    Create a `src` directory and the two files we will be working with.
    ```bash
    touch src/AgentMonad.ts
    touch src/main.ts
    ```

---

## 3. The MCE Core Library (`src/AgentMonad.ts`)

This file contains the core implementation of the `AgentMonad`, including its type definition and the API (`then`, `map`, `gather`, etc.) for creating and composing agentic workflows. It is the direct translation of the MCE framework from the paper into code.

```typescript
// src/AgentMonad.ts

import { pipe } from 'fp-ts/lib/function';
import *s from 'fp-ts/lib/StateT';
import *te from 'fp-ts/lib/TaskEither';
import *t from 'fp-ts/lib/Task';
import *a from 'fp-ts/lib/Array';

// --- Type Definition ---
// S: State, E: Error, A: Value
// AgentMonad<S, E, A> is a function:
// (state: S) => TaskEither<E, [A, S]>
// It takes the current state and returns a potentially failing asynchronous computation.
// On success, this computation yields a tuple of [newValue, newState].
export type AgentMonad<S, E, A> = s.StateT<S, te.TaskEither<E, any>, A>;

// --- MCE API ---

/**
 * The core sequential chaining operation (bind, flatMap, >>=).
 * It takes a function that receives the value from the previous step 
 * and returns a new AgentMonad, allowing workflows to be chained.
 * @param func The function to apply to the value, producing the next step in the flow.
 */
export const then = <S, E, A, B>(
    func: (value: A) => AgentMonad<S, E, B>
): ((monad: AgentMonad<S, E, A>) => AgentMonad<S, E, B>) => s.chain(func);

/**
 * Applies a pure function to the value inside the Monad, without altering the context.
 * (Functor's fmap)
 * @param func The pure function to apply to the wrapped value.
 */
export const map = <S, E, A, B>(
    func: (value: A) => B
): ((monad: AgentMonad<S, E, A>) => AgentMonad<S, E, B>) => s.map(func);

// --- Constructors & Helpers ---

/**
 * Begins a new MCE flow from an initial state.
 * The initial value is the state itself.
 */
export const start = <S, E, A>(initialState: S): AgentMonad<S, E, S> => {
    return s.of(initialState);
};

/**
 * Creates a step in the flow that is guaranteed to succeed with a given value.
 * @param value The value for the successful step.
 */
export const success = <S, E, A>(value: A): AgentMonad<S, E, A> => {
    return (state: S) => te.right([value, state]);
};

/**
 * Creates a step in the flow that is guaranteed to fail with a given error.
 * The entire monadic chain will short-circuit at this point.
 * @param error The error to fail with.
 */
export const failure = <S, E, A>(error: E): AgentMonad<S, E, A> => {
    return () => te.left(error);
};

/**
 * Applies a function to modify the current state.
 * @param modifier A function that takes the current state and returns the new state.
 */
export const modify = <S, E>(modifier: (s: S) => S): AgentMonad<S, E, void> => {
    return s.modify(modifier);
};

/**
 * Retrieves the current state from within the monadic flow.
 */
export const get = <S, E>(): AgentMonad<S, E, S> => s.get();

/**
 * Lifts a simple, fallible, asynchronous operation (a TaskEither) into the AgentMonad context.
 * This is the primary bridge for interacting with the outside world (e.g., APIs, file system).
 * @param task The TaskEither to lift.
 */
export const lift = <S, E, A>(task: te.TaskEither<E, A>): AgentMonad<S, E, A> => {
    return (state: S) => pipe(
        task,
        te.map(value => [value, state]) // On success, the value is wrapped, and the state is passed through unchanged.
    );
};

// --- Parallel Processing (Applicative) ---

/**
 * Executes an array of independent AgentMonads concurrently.
 * If any AgentMonad in the array fails, the entire gathered operation fails.
 * State management strategy: On success, the state from the *last* operation in the array is propagated.
 * @param tasks An array of AgentMonad instances to run in parallel.
 */
export const gather = <S, E, A>(
    tasks: AgentMonad<S, E, A>[]
): AgentMonad<S, E, A[]> => {
    return (state: S) => {
        // For each AgentMonad, apply the initial state to get a runnable TaskEither.
        const runners = tasks.map(task => task(state));

        // Use TaskEither's Applicative instance for parallel execution.
        const sequenced = pipe(
            runners,
            a.sequence(te.ApplicativePar)
        );

        return pipe(
            sequenced,
            te.map(results => {
                // `results` is an array of [value, newState] tuples.
                const finalState = results.length > 0 ? results[results.length - 1][1] : state;
                const finalValues = results.map(([value, _]) => value);
                return [finalValues, finalState];
            })
        );
    };
};

/**
 * Executes the entire monadic workflow with a given initial state.
 * This is the final step that triggers all the computations.
 * @param flow The complete AgentMonad workflow to run.
 * @param initialState The starting state for the agent.
 * @returns A Promise that resolves to an Either, containing the error or the final [value, state].
 */
export const run = <S, E, A>(flow: AgentMonad<S, E, A>, initialState: S): Promise<{ _tag: 'Left', left: E } | { _tag: 'Right', right: [A, S] }> => {
    return flow(initialState)(); // The final () triggers the TaskEither to run.
};
```

---

## 4. The Agent Implementation (`src/main.ts`)

This file demonstrates how to *use* the MCE library to build the "Simple Research Agent" from the paper. It shows how to define states, errors, and agentic steps, and how to chain them together into a resilient workflow that explicitly calls the `gemini-cli`.

### Prerequisites for Running
You must have `@google/gemini-cli` installed globally and configured with your API key.

```bash
# 1. Install the CLI
npm install -g @google/gemini-cli

# 2. Configure your API Key
gemini config set
```

### Implementation Code

```typescript
// src/main.ts

import { pipe } from 'fp-ts/lib/function';
import *E from 'fp-ts/lib/Either';
import *TE from 'fp-ts/lib/TaskEither';
import { spawn } from 'child_process'; // Use the safer 'spawn' for command execution

// Import our powerful MCE library!
import * as MCE from './AgentMonad';

// --- Define Agent's State and Error types ---
interface AgentState {
    task: string;
    history: string[];
}

// Define a richer set of possible errors for our agent.
type AgentError = 
    | 'ToolNotFound' 
    | 'SynthesisFailed' 
    | 'CLIExecutionFailed'
    | 'JSONParsingFailed';

// --- Dedicated, Safe Gemini CLI Invocation ---
// This function is our bridge to the real world. It explicitly calls the 'gemini' command.
const invokeGemini = (args: string[]): TE.TaskEither<AgentError, string> => {
    console.log(`\n�� Spawning: gemini ${args.join(' ')}`);

    // We wrap the callback/event-based `spawn` API in a Promise,
    // which can then be lifted into a TaskEither.
    return TE.tryCatch(
        () => new Promise<string>((resolve, reject) => {
            const process = spawn('gemini', args);
            let stdout = '';
            let stderr = '';

            process.stdout.on('data', (data) => { stdout += data.toString(); });
            process.stderr.on('data', (data) => { stderr += data.toString(); });
            process.on('close', (code) => {
                if (code === 0) {
                    resolve(stdout);
                } else {
                    reject(new Error(`Gemini CLI exited with code ${code}. Stderr: ${stderr}`));
                }
            });
            process.on('error', (err) => { reject(err); });
        }),
        (error: unknown) => {
            console.error(`CLI execution error:`, error);
            return 'CLIExecutionFailed'; // Map any error to our domain-specific error type.
        }
    );
}

// --- Define Agent's Behavioral Steps ---
// Each function returns an AgentMonad, encapsulating a single step of the workflow.

const planAction = (taskDescription: string): MCE.AgentMonad<AgentState, AgentError, string> => pipe(
    MCE.get<AgentState, AgentError>(), // Get the current state
    MCE.then(state => {
        const prompt = `Create a simple, one-step plan to answer the question: "${state.task}". The plan should involve using a 'search' tool. Respond with only the plan string.`;
        const args = ['gen', prompt];

        return pipe(
            MCE.lift(invokeGemini(args)), // Lift the real-world action into the monad
            MCE.then(plan => pipe( // If successful, update the state
                MCE.modify<AgentState, AgentError>(s => ({...s, history: [...s.history, `Plan: ${plan.trim()}`]})),
                MCE.then(() => MCE.success(plan.trim())) // And return the plan as the new value
            ))
        );
    })
);

const executeTool = (plan: string): MCE.AgentMonad<AgentState, AgentError, any> => pipe(
    MCE.get<AgentState, AgentError>(),
    MCE.then(state => {
        if (plan.toLowerCase().includes("search")) {
            const prompt = `You are a search tool. Fulfill this task: "${state.task}". Respond with a JSON object like {"result": "your findings"}.`;
            const args = ['gen', prompt, '--json'];
            
            return pipe(
                MCE.lift(invokeGemini(args)),
                // Chain another operation to parse the JSON output
                MCE.then(stdout => {
                    // Parsing is also a fallible operation, so we wrap it in an Either.
                    const parsed = E.tryCatch(
                        () => JSON.parse(stdout),
                        () => 'JSONParsingFailed' as AgentError
                    );

                    // Based on whether parsing succeeded or failed, return a new AgentMonad
                    return pipe(
                        parsed,
                        E.match(
                            (error) => MCE.failure(error), // On parsing failure, short-circuit the whole flow
                            (jsonValue) => pipe( // On success, update history and return the JSON value
                                MCE.modify<AgentState, AgentError>(s => ({ ...s, history: [...s.history, `Tool Output: ${JSON.stringify(jsonValue)}`] })),
                                MCE.then(() => MCE.success(jsonValue))
                            )
                        )
                    );
                })
            );
        } else {
            return MCE.failure('ToolNotFound');
        }
    })
);

const synthesizeAnswer = (toolOutput: any): MCE.AgentMonad<AgentState, AgentError, string> => pipe(
    MCE.get<AgentState, AgentError>(),
    MCE.then(state => {
        const prompt = `Based on this JSON data: ${JSON.stringify(toolOutput)}, write a final, user-friendly answer for the task: "${state.task}"`;
        const args = ['gen', prompt];
        
        return pipe(
            MCE.lift(invokeGemini(args)),
            MCE.then(finalAnswer => pipe(
                MCE.modify<AgentState, AgentError>(s => ({...s, history: [...s.history, `Final Answer: ${finalAnswer.trim()}`]})),
                MCE.then(() => MCE.success(finalAnswer.trim()))
            ))
        );
    })
);

// --- Orchestrate and Run the Agent ---

const main = async () => {
    console.log("�� STARTING: Flow with REAL `gemini-cli` calls via a dedicated invoker.");
    const initialState: AgentState = { task: "Explain Monadic Context Engineering in simple terms", history: [] };

    // The agent's entire logic is a declarative, linear chain of operations.
    // It reads like a high-level description of the workflow.
    const agentFlow = pipe(
        MCE.start<AgentState, AgentError, AgentState>(initialState),
        MCE.then(state => planAction(state.task)),
        MCE.then(executeTool),
        MCE.then(synthesizeAnswer),
        MCE.map(answer => `✅ FINAL REPORT:\n${answer}`) // map is used for the final, pure transformation
    );

    // Trigger the execution and wait for the result.
    const result = await MCE.run(agentFlow, initialState);

    // Handle the final result, whether it's a success or a failure.
    pipe(
        result,
        E.match(
            (error) => console.error(`\n�� FLOW FAILED: ${error}`),
            ([value, state]) => {
                console.log(`\n\n✨ FLOW SUCCEEDED! ✨`);
                console.log(value);
                console.log("\n--- Execution History ---");
                state.history.forEach(h => console.log(`- ${h}`));
            }
        )
    );
};

main();
```

---

## 5. How to Run

Navigate to your project's root directory in the terminal and execute the following command:

```bash
npx ts-node src/main.ts
```

---

## 6. Expected Output

You will see the logs from the `invokeGemini` function, followed by the final result of the monadic flow. The output will look similar to this (Gemini's responses may vary):

```
�� STARTING: Flow with REAL `gemini-cli` calls via a dedicated invoker.

�� Spawning: gemini gen Create a simple, one-step plan to answer the question: "Explain Monadic Context Engineering in simple terms". The plan should involve using a 'search' tool. Respond with only the plan string.

�� Spawning: gemini gen You are a search tool. Fulfill this task: "Explain Monadic Context Engineering in simple terms". Respond with a JSON object like {"result": "your findings"}. --json

�� Spawning: gemini gen Based on this JSON data: {"result":"Monadic Context Engineering is a functional programming paradigm for building robust AI agents. It uses monads to manage state, errors, and side-effects in a structured way."}, write a final, user-friendly answer for the task: "Explain Monadic Context Engineering in simple terms"


✨ FLOW SUCCEEDED! ✨
✅ FINAL REPORT:
Monadic Context Engineering (MCE) is a powerful method for building reliable and predictable AI agents. Instead of writing complex, error-prone code, MCE uses a functional programming concept called a "monad" to cleanly manage an agent's entire workflow. Think of it as a railway system for data and actions, automatically handling state changes, potential errors, and interactions with the outside world, allowing developers to focus on the agent's core logic.

--- Execution History ---
- Plan: Use the 'search' tool to find a simple definition of Monadic Context Engineering.
- Tool Output: {"result":"Monadic Context Engineering is a functional programming paradigm for building robust AI agents. It uses monads to manage state, errors, and side-effects in a structured way."}
- Final Answer: Monadic Context Engineering (MCE) is a powerful method for building reliable and predictable AI agents. Instead of writing complex, error-prone code, MCE uses a functional programming concept called a "monad" to cleanly manage an agent's entire workflow. Think of it as a railway system for data and actions, automatically handling state changes, potential errors, and interactions with the outside world, allowing developers to focus on the agent's core logic.
```

---

## 7. Architectural Benefits

This implementation demonstrates the key advantages of Monadic Context Engineering:

- **Declarative & Legible**: The main workflow is a clear, linear sequence of steps.
- **Robust by Default**: Errors from CLI calls or JSON parsing are automatically caught and short-circuit the flow without `try/catch` boilerplate.
- **Maximal Composability**: Each step (`planAction`, `executeTool`) is a self-contained, verifiable unit that can be easily reused, reordered, or replaced.
- **Principled Side-Effect Management**: All interactions with the "real world" (the `gemini-cli`) are explicitly lifted into the monadic context, cleanly separating pure logic from impure actions.

This framework, based on the principles from the MCE paper, provides a solid foundation for building the next generation of reliable and scalable AI agents.

