// src/Monadic.ts

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
