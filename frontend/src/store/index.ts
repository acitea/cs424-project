import { atom } from 'jotai';

// Define interfaces for our state
export interface ImageTask {
  id: string;
  name: string;
  description: string;
}

export interface TaskState {
  file: File | null;
  preview: string | null;
  result: string | null;
  loading: boolean;
  progress: number;
}

// Define initial task configurations
export const tasksAtom = atom<ImageTask[]>([
  { 
    id: 'task1', 
    name: 'Task 1',
    description: 'Transform your images using our first CycleGAN model. This model specializes in converting between two specific domains.'
  },
  { 
    id: 'task2', 
    name: 'Task 2',
    description: 'Create cute pokemon from your favourite animals using our custom CycleGAN!'
  },
]);

// State for active task ID
export const activeTaskIdAtom = atom<string>('task1');

// Task state storage - persists across tab switches
export const taskStatesAtom = atom<Record<string, TaskState>>({
  task1: {
    file: null,
    preview: null,
    result: null,
    loading: false,
    progress: 0
  },
  task2: {
    file: null,
    preview: null,
    result: null,
    loading: false,
    progress: 0
  }
});

// Derived atom for the current task
export const currentTaskAtom = atom(
  (get) => {
    const taskId = get(activeTaskIdAtom);
    const tasks = get(tasksAtom);
    return tasks.find(task => task.id === taskId) || tasks[0];
  }
);

// Derived atom for the current task state
export const currentTaskStateAtom = atom(
  (get) => {
    const taskId = get(activeTaskIdAtom);
    const taskStates = get(taskStatesAtom);
    return taskStates[taskId];
  }
);

// Setter atoms for updating specific task states
export const updateTaskFileAtom = atom(
  null,
  (get, set, { taskId, file }: { taskId: string; file: File | null }) => {
    const taskStates = get(taskStatesAtom);
    set(taskStatesAtom, {
      ...taskStates,
      [taskId]: {
        ...taskStates[taskId],
        file,
      }
    });
  }
);

export const updateTaskPreviewAtom = atom(
  null,
  (get, set, { taskId, preview }: { taskId: string; preview: string | null }) => {
    const taskStates = get(taskStatesAtom);
    set(taskStatesAtom, {
      ...taskStates,
      [taskId]: {
        ...taskStates[taskId],
        preview,
      }
    });
  }
);

export const updateTaskResultAtom = atom(
  null,
  (get, set, { taskId, result }: { taskId: string; result: string | null }) => {
    const taskStates = get(taskStatesAtom);
    set(taskStatesAtom, {
      ...taskStates,
      [taskId]: {
        ...taskStates[taskId],
        result,
      }
    });
  }
);

export const updateTaskLoadingAtom = atom(
  null,
  (get, set, { taskId, loading }: { taskId: string; loading: boolean }) => {
    const taskStates = get(taskStatesAtom);
    set(taskStatesAtom, {
      ...taskStates,
      [taskId]: {
        ...taskStates[taskId],
        loading,
      }
    });
  }
);

export const updateTaskProgressAtom = atom(
  null,
  (get, set, { taskId, progress }: { taskId: string; progress: number }) => {
    const taskStates = get(taskStatesAtom);
    set(taskStatesAtom, {
      ...taskStates,
      [taskId]: {
        ...taskStates[taskId],
        progress,
      }
    });
  }
);