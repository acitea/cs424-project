import React from 'react';
import { useAtom } from 'jotai';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Separator } from '@/components/ui/separator';
import { Button } from '@/components/ui/button';
import ImageUploader from './ImageUploader';
import ImageDisplay from './ImageDisplay';
import LoadingIndicator from './LoadingIndicator';
import { 
  activeTaskIdAtom, 
  tasksAtom, 
  taskStatesAtom,
  updateTaskResultAtom,
  updateTaskLoadingAtom,
  updateTaskProgressAtom
} from '@/store';

const ImageGenerationSection: React.FC = () => {
  const [tasks] = useAtom(tasksAtom);
  const [activeTaskId, setActiveTaskId] = useAtom(activeTaskIdAtom);
  const [taskStates] = useAtom(taskStatesAtom);
  const [, updateTaskResult] = useAtom(updateTaskResultAtom);
  const [, updateTaskLoading] = useAtom(updateTaskLoadingAtom);
  const [, updateTaskProgress] = useAtom(updateTaskProgressAtom);

  const handleSubmit = async (taskId: string) => {
    const taskState = taskStates[taskId];
    
    if (!taskState.file) return;

    // Update loading state
    updateTaskLoading({ taskId, loading: true });
    updateTaskProgress({ taskId, progress: 0 });

    // Simulate progress updates
    const progressInterval = setInterval(() => {
      updateTaskProgress({ 
        taskId, 
        progress: Math.min(taskStates[taskId].progress + 10, 95) 
      });
    }, 500);

    try {
      // mock first
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // mock first
      updateTaskResult({ 
        taskId, 
        result: 'https://placehold.co/600x400/png' 
      });
    } catch (error) {
      console.error('Error generating image:', error);
      toast.error('Failed to generate image');
    } finally {
      clearInterval(progressInterval);
      updateTaskProgress({ taskId, progress: 100 });
      setTimeout(() => {
        updateTaskLoading({ taskId, loading: false });
      }, 500);
    }
  };

  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-6">Image Generation Tasks</h1>
      
      <Tabs 
        defaultValue="task1" 
        value={activeTaskId} 
        onValueChange={setActiveTaskId}
        className="w-full"
      >
        <TabsList className="w-full max-w-md mb-8 grid grid-cols-2 p-1 bg-muted/80">
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
              
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-medium mb-3">Input</h3>
                  <ImageUploader 
                    label="Upload Source Image"
                    taskId={task.id}
                    previewImage={taskState.preview}
                  />
                </div>
                
                <div>
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
                    message="Generating image..."
                    className="max-w-md"
                  />
                ) : (
                  <Button
                    onClick={() => handleSubmit(task.id)}
                    disabled={!taskState.file}
                    className="w-40 btn-primary"
                  >
                    Generate Image
                  </Button>
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