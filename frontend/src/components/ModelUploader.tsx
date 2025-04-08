import React, { useState } from 'react';
import { Upload, X, Check, Loader2 } from 'lucide-react';
import { toast } from 'sonner';
import { cn } from '@/utils';

interface ModelUploaderProps {
  taskId: string;
  reverse: boolean;
  onUploadSuccess?: (model: any) => void;
  disabled?: boolean;
  className?: string;
  apiUrl?: string;
}

const ModelUploader: React.FC<ModelUploaderProps> = ({
  taskId,
  reverse,
  onUploadSuccess,
  disabled = false,
  className,
  apiUrl = '/api'
}) => {
  const [uploading, setUploading] = useState(false);
  const [modelName, setModelName] = useState('');
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [showForm, setShowForm] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      if (!file.name.endsWith('.pth')) {
        toast.error('Only .pth model files are supported');
        return;
      }
      setModelFile(file);
      // Extract name without extension for the model name field
      const nameWithoutExt = file.name.replace(/\.pth$/, '');
      setModelName(nameWithoutExt);
    }
  };

  const handleUpload = async () => {
    if (!modelFile) {
      toast.error('Please select a model file');
      return;
    }

    setUploading(true);

    try {
      const formData = new FormData();
      formData.append('model_file', modelFile);
      formData.append('reverse', reverse.toString());
      if (modelName) {
        formData.append('model_name', modelName);
      }

      const response = await fetch(`${apiUrl}/upload-model/${taskId}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Failed to upload model' }));
        throw new Error(errorData.detail || 'Failed to upload model');
      }

      const data = await response.json();
      toast.success('Model uploaded successfully');

      // Reset the form
      setModelFile(null);
      setModelName('');
      setShowForm(false);

      // Notify parent component
      if (onUploadSuccess) {
        onUploadSuccess(data.model);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to upload model';
      toast.error(errorMessage);
    } finally {
      setUploading(false);
    }
  };

  const handleCancel = () => {
    setShowForm(false);
    setModelFile(null);
    setModelName('');
  };

  return (
    <div className={cn("relative", className)}>
      {!showForm ? (
        <button
          type="button"
          onClick={() => setShowForm(true)}
          disabled={disabled}
          className={cn(
            "flex items-center gap-1 text-xs py-1 px-2 rounded border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors",
            "text-gray-700 dark:text-gray-200",
            disabled && "opacity-50 cursor-not-allowed"
          )}
        >
          <Upload size={14} />
          <span>Upload Model</span>
        </button>
      ) : (
        <div className="absolute right-0 top-0 z-20 w-72 p-3 rounded-md shadow-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
          <div className="flex justify-between items-center mb-2">
            <h3 className="text-sm font-medium">Upload New Model</h3>
            <button
              type="button"
              onClick={handleCancel}
              className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              <X size={16} />
            </button>
          </div>
          
          <div className="space-y-3">
            <div>
              <label className="block text-xs text-gray-700 dark:text-gray-300 mb-1">
                Model File (.pth)
              </label>
              <div className="flex items-center">
                <input
                  type="file"
                  accept=".pth"
                  onChange={handleFileChange}
                  className="sr-only"
                  id="model-file-input"
                  disabled={uploading}
                />
                <label
                  htmlFor="model-file-input"
                  className={cn(
                    "flex-1 cursor-pointer flex items-center justify-center gap-1 text-xs py-1.5 px-2 rounded border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors",
                    uploading && "opacity-50 cursor-not-allowed"
                  )}
                >
                  <Upload size={14} />
                  <span className="truncate">
                    {modelFile ? modelFile.name : 'Choose file'}
                  </span>
                </label>
              </div>
            </div>
            
            <div>
              <label htmlFor="model-name" className="block text-xs text-gray-700 dark:text-gray-300 mb-1">
                Model Name (optional)
              </label>
              <input
                type="text"
                id="model-name"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                placeholder="Custom name for the model"
                className="w-full text-xs py-1.5 px-2 rounded border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900"
                disabled={uploading}
              />
            </div>
            
            <div className="flex items-center text-xs">
              <span className="mr-1">Direction:</span>
              <span className={cn(
                "font-medium",
                reverse ? "text-purple-600 dark:text-purple-400" : "text-blue-600 dark:text-blue-400"
              )}>
                {reverse ? 'Reverse' : 'Forward'}
              </span>
            </div>
            
            <div className="flex justify-end gap-2 pt-1">
              <button
                type="button"
                onClick={handleCancel}
                className="text-xs py-1.5 px-3 rounded border border-gray-300 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700"
                disabled={uploading}
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleUpload}
                disabled={!modelFile || uploading}
                className={cn(
                  "flex items-center gap-1 text-xs py-1.5 px-3 rounded border border-transparent bg-blue-600 text-white hover:bg-blue-700",
                  "disabled:bg-blue-600/50 disabled:cursor-not-allowed"
                )}
              >
                {uploading ? (
                  <>
                    <Loader2 size={14} className="animate-spin" />
                    <span>Uploading...</span>
                  </>
                ) : (
                  <>
                    <Check size={14} />
                    <span>Upload</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelUploader;