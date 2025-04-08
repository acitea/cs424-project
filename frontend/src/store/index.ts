import { atom } from 'jotai';

// Define types
export interface Task {
  id: string;
  name: string;
  description: string;
  conversion_key?: string;
}

export interface Model {
  id: string;
  name: string;
  description: string;
  reverse: boolean;
}

export interface TaskState {
  file: File | null;
  preview: string | null;
  result: string | null;
  loading: boolean;
  progress: number;
  error: string | null;
  reverse: boolean | null;
  selectedModel: Model | null;
}

// Define the initial task state
const initialTaskState: TaskState = {
  file: null,
  preview: null,
  result: null,
  loading: false,
  progress: 0,
  error: null,
  reverse: false,
  selectedModel: null
};

// API URL
export const apiUrlAtom = atom<string>('http://localhost:8000');

// Available tasks

export const tasksAtom = atom<Task[]>([
  {
    id: 'task1',
    name: 'Style Transfer',
    description: 'Transform your photos with artistic style effects'
  },
  {
    id: 'task2',
    name: 'Pokemon Generator',
    description: 'Turn your image into a Pokemon-style character'
  }
]);


// Active task ID
export const activeTaskIdAtom = atom<string>('task1');

// Task states - a map of task IDs to their respective states
export const taskStatesAtom = atom<Record<string, TaskState>>({
  task1: { ...initialTaskState },
  task2: { ...initialTaskState },
});

// Derived atoms for updating specific parts of the task state
export const updateTaskFileAtom = atom(
  null,
  (get, set, { taskId, file}: { taskId: string; file: File | null;}) => {
    set(taskStatesAtom, {
      ...get(taskStatesAtom),
      [taskId]: {
        ...get(taskStatesAtom)[taskId],
        file,
      },
    });
  }
);

export const updateTaskReverseAtom = atom(
  null,
  (get, set, { taskId, reverse}: { taskId: string; reverse: boolean | null;}) => {
    set(taskStatesAtom, {
      ...get(taskStatesAtom),
      [taskId]: {
        ...get(taskStatesAtom)[taskId],
        reverse,
      },
    });
  }
);

export const updateTaskPreviewAtom = atom(
  null,
  (get, set, { taskId, preview }: { taskId: string; preview: string | null }) => {
    set(taskStatesAtom, {
      ...get(taskStatesAtom),
      [taskId]: {
        ...get(taskStatesAtom)[taskId],
        preview,
      },
    });
  }
);

export const updateTaskSelectedModelAtom = atom(
  null,
  (get, set, update: { taskId: string; model: Model | null }) => {
    const { taskId, model } = update;
    const taskStates = get(taskStatesAtom);
    
    // Also update the reverse flag based on the selected model
    const reverse = model ? model.reverse : false;
    
    set(taskStatesAtom, {
      ...taskStates,
      [taskId]: {
        ...taskStates[taskId],
        selectedModel: model,
        reverse
      }
    });
  }
);

export const updateTaskResultAtom = atom(
  null,
  (get, set, { taskId, result }: { taskId: string; result: string | null }) => {
    set(taskStatesAtom, {
      ...get(taskStatesAtom),
      [taskId]: {
        ...get(taskStatesAtom)[taskId],
        result,
      },
    });
  }
);

export const updateTaskLoadingAtom = atom(
  null,
  (get, set, { taskId, loading }: { taskId: string; loading: boolean }) => {
    set(taskStatesAtom, {
      ...get(taskStatesAtom),
      [taskId]: {
        ...get(taskStatesAtom)[taskId],
        loading,
      },
    });
  }
);

export const updateTaskProgressAtom = atom(
  null,
  (get, set, { taskId, progress }: { taskId: string; progress: number }) => {
    set(taskStatesAtom, {
      ...get(taskStatesAtom),
      [taskId]: {
        ...get(taskStatesAtom)[taskId],
        progress,
      },
    });
  }
);

export const updateTaskErrorAtom = atom(
  null,
  (get, set, { taskId, error }: { taskId: string; error: string | null }) => {
    set(taskStatesAtom, {
      ...get(taskStatesAtom),
      [taskId]: {
        ...get(taskStatesAtom)[taskId],
        error,
      },
    });
  }
);