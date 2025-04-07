import { useCallback, HTMLAttributes } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X } from 'lucide-react';
import { toast } from 'sonner';
import { cn, isImageFile, createImagePreview } from '@/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { useAtom } from 'jotai';
import { updateTaskFileAtom, updateTaskPreviewAtom } from '@/store';

interface ImageUploaderProps extends HTMLAttributes<HTMLDivElement> {
  label?: string;
  taskId: string;
  previewImage: string | null;
}

const ImageUploader: React.FC<ImageUploaderProps> = ({ 
  label = 'Upload Image', 
  taskId,
  previewImage,
  className, 
  ...props 
}) => {
  const [, updateTaskFile] = useAtom(updateTaskFileAtom);
  const [, updateTaskPreview] = useAtom(updateTaskPreviewAtom);
  
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    try {
      if (acceptedFiles.length === 0) {
        return;
      }
      
      const file = acceptedFiles[0];
      
      if (!isImageFile(file)) {
        const error = 'Please upload an image file (JPG, PNG, etc.)';
        toast.error(error);
        throw new Error(error);
      }
      
      if (file.size > 5 * 1024 * 1024) {
        const error = 'File size exceeds 5MB limit';
        toast.error(error);
        throw new Error(error);
      }
      
      const previewUrl = await createImagePreview(file);
      
      // Update the task state
      updateTaskFile({ taskId, file });
      updateTaskPreview({ taskId, preview: previewUrl });
      
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';
      toast.error(errorMessage);
      console.error('Error in image upload:', err);
    }
  }, [taskId, updateTaskFile, updateTaskPreview]);
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.webp']
    },
    maxFiles: 1
  });
  
  const handleClearImage = (e: React.MouseEvent<HTMLButtonElement>) => {
    e.stopPropagation();
    updateTaskFile({ taskId, file: null });
    updateTaskPreview({ taskId, preview: null });
  };
  
  return (
    <Card className={cn('w-full', className)} {...props}>
      <CardHeader className="pb-3">
        <CardTitle className="text-lg">{label}</CardTitle>
      </CardHeader>
      
      <CardContent>
        <div
          {...getRootProps()}
          className={cn(
            "border-2 border-dashed rounded-md p-6 text-center cursor-pointer transition-colors duration-200",
            isDragActive ? "border-primary bg-primary/5" : "border-muted",
            !previewImage && "hover:border-primary hover:bg-primary/5"
          )}
        >
          <input {...getInputProps()} />
          
          {previewImage ? (
            <div className="relative">
              <img src={previewImage} alt="Preview" className="mx-auto max-h-48 object-contain" />
              <Button
                onClick={handleClearImage}
                variant="destructive"
                size="icon"
                className="absolute top-0 right-0 h-6 w-6 p-1 rounded-full shadow-md transform translate-x-1/2 -translate-y-1/2"
                aria-label="Remove image"
                type="button"
              >
                <X size={14} />
              </Button>
            </div>
          ) : (
            <div className="text-center">
              <Upload className="mx-auto h-12 w-12 text-muted-foreground" aria-hidden="true" />
              <p className="mt-2 text-sm text-foreground">
                {isDragActive ? 'Drop the image here' : 'Drag & drop an image, or click to select'}
              </p>
              <p className="mt-1 text-xs text-muted-foreground">
                PNG, JPG, GIF up to 5MB
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default ImageUploader;