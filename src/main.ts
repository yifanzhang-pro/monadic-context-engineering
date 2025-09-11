// src/main.ts

import { pipe } from 'fp-ts/lib/function';
import *E from 'fp-ts/lib/Either';
import *TE from 'fp-ts/lib/TaskEither';
import { spawn } from 'child_process'; // Use the safer 'spawn' for command execution

// Import our powerful MCE library!
import * as MCE from './Monadic';

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
    console.log(`\n Spawning: gemini ${args.join(' ')}`);

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
    console.log(" STARTING: Flow with REAL `gemini-cli` calls via a dedicated invoker.");
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
            (error) => console.error(`\n FLOW FAILED: ${error}`),
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
