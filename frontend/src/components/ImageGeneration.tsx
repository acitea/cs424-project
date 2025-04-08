import React, { useState, useEffect } from 'react';
import { useAtom } from 'jotai';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { Button } from '@/components/ui/button';
import { toast } from 'sonner';
import ImageUploader from './ImageUploader';
import ImageDisplay from './ImageDisplay';
import LoadingIndicator from './LoadingIndicator';
import ReverseButton from './ReverseButton';
import { 
  activeTaskIdAtom, 
  tasksAtom, 
  taskStatesAtom,
  updateTaskResultAtom,
  updateTaskLoadingAtom,
  updateTaskProgressAtom,
  updateTaskErrorAtom,
  updateTaskReverseAtom,
  updateTaskPreviewAtom,
  apiUrlAtom,
  Task,
  TaskState
} from '@/store';

const ImageGenerationSection: React.FC = () => {
  const [tasks, setTasks] = useAtom(tasksAtom);
  const [activeTaskId, setActiveTaskId] = useAtom(activeTaskIdAtom);
  const [taskStates, setTaskStates] = useAtom(taskStatesAtom);
  const [apiUrl] = useAtom(apiUrlAtom);
  const [, updateTaskResult] = useAtom(updateTaskResultAtom);
  const [, updateTaskLoading] = useAtom(updateTaskLoadingAtom);
  const [, updateTaskProgress] = useAtom(updateTaskProgressAtom);
  const [, updateTaskError] = useAtom(updateTaskErrorAtom);
  const [, updateTaskReverse] = useAtom(updateTaskReverseAtom);
  const [, updateTaskPreview] = useAtom(updateTaskPreviewAtom);
  const [processingTask, setProcessingTask] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch tasks from the backend
  useEffect(() => {
    const fetchTasks = async () => {
      setIsLoading(true);
      setError(null);
      
      try {
        const response = await fetch(`${apiUrl}/tasks`);
        if (!response.ok) {
          throw new Error('Failed to fetch tasks');
        }
        
        const data = await response.json();
        
        if (data.tasks && Array.isArray(data.tasks)) {
          setTasks(data.tasks);
          
          // Initialize task states for new tasks
          const newTaskStates: Record<string, TaskState> = {};
          
          data.tasks.forEach((task: Task) => {
            newTaskStates[task.id] = taskStates[task.id] || {
              file: null,
              preview: null,
              result: null,
              loading: false,
              progress: 0,
              error: null,
              reverse: false,
            };
          });
          
          setTaskStates(newTaskStates);
          
          if (data.tasks.length > 0 && !activeTaskId) {
            setActiveTaskId(data.tasks[0].id);
          }
        } else {
          throw new Error('Invalid response format');
        }
      } catch (error) {
        console.error('Error fetching tasks:', error);
        setError(error instanceof Error ? error.message : 'An unknown error occurred');
        toast.error('Failed to fetch tasks');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchTasks();
  }, [apiUrl]);

  const handleSubmit = async (taskId: string) => {
    const taskState = taskStates[taskId];
    
    if (!taskState.file) {
      toast.error('Please upload an image first');
      return;
    }

    // Update loading state
    updateTaskLoading({ taskId, loading: true });
    updateTaskProgress({ taskId, progress: 0 });
    updateTaskError({ taskId, error: null });
    updateTaskReverse({ taskId, reverse: taskState.reverse });
    setProcessingTask(taskId);

    // Create form data
    const formData = new FormData();
    formData.append('file', taskState.file);
    console.log('taskState.reverse upon submit:', taskState.reverse);
    const reverseParam = taskState.reverse ? 'true' : 'false';

    try {
      // Start progress simulation
      const progressInterval = setInterval(() => {
        updateTaskProgress({ 
          taskId, 
          progress: Math.min(taskStates[taskId].progress + 5, 95) 
        });
      }, 300);

      // Make API request
      const response = await fetch(`${apiUrl}/transform/${taskId}?reverse=${reverseParam}`, {
        method: 'POST',
        body: formData,
      });

      // Clear progress simulation
      clearInterval(progressInterval);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to transform image');
      }

      const data = await response.json();
      
      // Complete progress
      updateTaskProgress({ taskId, progress: 100 });
      
      // Wait a moment to show completed progress
      setTimeout(() => {
        // Update with result
        updateTaskResult({ 
          taskId, 
          result: `${apiUrl}${data.result_url}` 
        });
        
        // Reset loading state
        updateTaskLoading({ taskId, loading: false });
        setProcessingTask(null);
        
        toast.success('Image transformed successfully!');
      }, 500);
      
    } catch (error) {
      console.error('Error transforming image:', error);
      
      let errorMessage = 'Failed to transform image';
      if (error instanceof Error) {
        errorMessage = error.message;
      }
      
      updateTaskError({ taskId, error: errorMessage });
      updateTaskLoading({ taskId, loading: false });
      setProcessingTask(null);
      toast.error(errorMessage);
    }
  };

  // Show loading state while fetching tasks
  if (isLoading) {
    return (
      <div className="container mx-auto py-8 flex items-center justify-center h-[400px]">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-lg">Loading image transformation options...</p>
        </div>
      </div>
    );
  }

  // Show error state if there was a problem
  if (error || tasks.length === 0) {
    return (
      <div className="container mx-auto py-8">
        <div className="bg-destructive/10 border border-destructive text-destructive p-6 rounded-lg">
          <h2 className="text-2xl font-semibold mb-4">Error Loading Tasks</h2>
          <p className="mb-4">{error || 'No tasks available'}</p>
          <Button 
            onClick={() => window.location.reload()}
            variant="outline"
            className="border-destructive text-destructive hover:bg-destructive/10"
          >
            Reload Page
          </Button>
        </div>
      </div>
    );
  }

  const handleReverse = (taskId: string) => {
    const taskState = taskStates[taskId];

    const tempPreview = taskState.preview;
    const tempResult = taskState.result;
    
    
    updateTaskResult({ taskId, result: tempResult });
    updateTaskPreview({ taskId, preview: tempPreview });
    updateTaskReverse({ taskId, reverse: !taskState.reverse });
    console.log('Reversing input and output images');
    console.log('reverse:', taskState.reverse);
    toast.success('Input and output images reversed');
  };

  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-6">Image Transformation Studio</h1>
      
      <Tabs 
        defaultValue={tasks[0]?.id || "task1"} 
        value={activeTaskId} 
        onValueChange={setActiveTaskId}
        className="w-full"
      >
        <TabsList className="w-full max-w-md mb-8 grid" style={{ gridTemplateColumns: `repeat(${tasks.length}, minmax(0, 1fr))` }}>
          {tasks.map(task => (
            <TabsTrigger 
              key={task.id} 
              value={task.id}
              className="data-[state=active]:bg-background data-[state=active]:text-primary data-[state=active]:border-b-2 data-[state=active]:border-primary rounded-none"
            >
              {task.name}
            </TabsTrigger>
          ))}
        </TabsList>
        
        {tasks.map(task => {
          const taskState = taskStates[task.id];
          
          return (
            <TabsContent 
              key={task.id} 
              value={task.id} 
              className="space-y-6 animate-in fade-in-100 duration-300"
            >
              <h2 className="text-2xl font-semibold mb-4">{task.name}</h2>
              <p className="text-muted-foreground mb-6">
                {task.description}
              </p>
              
              <div className="flex flex-row items-center justify-between">
                <div className="flex flex-col items-center w-[45%]">
                  <h3 className="text-lg font-medium mb-3">Input</h3>
                  <ImageUploader 
                    label="Upload Source Image"
                    taskId={task.id}
                    previewImage={taskState.preview}
                    disabled={taskState.loading}
                  />
                </div>
                
                <div className="flex flex-col items-center justify-center gap-2">
                  <ReverseButton 
                    onClick={() => handleReverse(task.id)}
                    // disabled={taskState.loading || !taskState.preview || !taskState.result}
                  />
                  <p>{taskState.reverse ? "B->A" : "A->B"}</p>
                </div>

                <div className="flex flex-col items-center w-[45%]">
                  <h3 className="text-lg font-medium mb-3">Output</h3>
                  <ImageDisplay 
                    title="Generated Image"
                    image={taskState.result}
                  />
                </div>
              </div>
              
              <Separator className="my-6 bg-muted-foreground/20" />
              
              <div className="flex flex-col items-center space-y-4">
                {taskState.loading ? (
                  <LoadingIndicator 
                    progress={taskState.progress} 
                    message={processingTask === task.id ? "Transforming image..." : "Starting..."}
                    className="max-w-md"
                  />
                ) : (
                  <Button
                    onClick={() => handleSubmit(task.id)}
                    disabled={!taskState.file}
                    className="w-40 btn-primary"
                  >
                    Transform Image
                  </Button>
                )}
                
                {taskState.error && (
                  <p className="text-destructive text-sm mt-2">{taskState.error}</p>
                )}
              </div>
            </TabsContent>
          );
        })}
      </Tabs>
    </div>
  );
};

export default ImageGenerationSection;