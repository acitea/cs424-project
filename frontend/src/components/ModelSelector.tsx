import React, { useEffect, useState, useRef } from 'react';
import { Loader2, ChevronDown } from 'lucide-react';
import { cn } from '@/utils';

interface Model {
  id: string;
  name: string;
  description: string;
  reverse: boolean;
}

interface ModelSelectorProps {
  taskId: string;
  selectedModelId: string;
  onModelSelect: (model: Model) => void;
  reverse: boolean;
  disabled?: boolean;
  className?: string;
  apiUrl?: string;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({
  taskId,
  selectedModelId,
  onModelSelect,
  disabled = false,
  className,
  apiUrl = '/api',
  reverse
}) => {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Fetch models
  useEffect(() => {
    const fetchModels = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response = await fetch(`${apiUrl}/available-models/${taskId}?reverse=${reverse}`);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch models: ${response.statusText}`);
        }
        
        const data = await response.json();
        setModels(data.models);
        
        // Auto-select the first model if none is selected
        if (data.models.length > 0 && (!selectedModelId || !data.models.find(m => m.id === selectedModelId))) {
          onModelSelect(data.models[0]);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load models');
        console.error('Error fetching models:', err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchModels();
  }, [taskId, apiUrl, selectedModelId, onModelSelect, reverse]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  // Handle model selection
  const handleModelSelect = (model: Model) => {
    onModelSelect(model);
    setIsOpen(false);
  };

  // Get currently selected model name
  const getSelectedModelName = () => {
    const model = models.find(m => m.id === selectedModelId);
    return model ? model.name : 'Select a model file';
  };

  // Toggle dropdown
  const toggleDropdown = () => {
    if (!disabled && !loading) {
      setIsOpen(!isOpen);
    }
  };

  return (
    <div className={cn("flex items-center", className)}>
      <div className="mr-2 text-sm font-medium whitespace-nowrap">Model:</div>
      <div className="w-64 relative" ref={dropdownRef}>
        {loading ? (
          <div className="flex items-center text-xs text-muted-foreground py-1.5">
            <Loader2 className="h-3 w-3 animate-spin mr-1" />
            <span>Loading...</span>
          </div>
        ) : error ? (
          <div className="text-xs text-destructive py-1.5">{error}</div>
        ) : models.length === 0 ? (
          <div className="text-xs text-muted-foreground py-1.5">No models available</div>
        ) : (
          <>
            {/* Dropdown Trigger */}
            <button
              type="button"
              onClick={toggleDropdown}
              disabled={disabled}
              className={cn(
                "flex h-8 w-full items-center justify-between rounded-md border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-3 py-1 text-sm text-left shadow-sm transition-colors",
                "hover:bg-gray-50 dark:hover:bg-gray-700",
                "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50",
                "disabled:cursor-not-allowed disabled:opacity-50"
              )}
              aria-haspopup="listbox"
              aria-expanded={isOpen}
            >
              <span className="truncate">{getSelectedModelName()}</span>
              <ChevronDown 
                className={cn(
                  "h-4 w-4 text-gray-500 dark:text-gray-400 transform transition-transform duration-200",
                  isOpen ? "rotate-180" : "rotate-0"
                )} 
              />
            </button>
            
            {/* Dropdown Menu */}
            {isOpen && (
              <div 
                className={cn(
                  "absolute z-10 mt-1 w-full rounded-md border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-lg",
                  "animate-in fade-in-20 zoom-in-95 duration-100",
                  "max-h-60 overflow-auto p-1"
                )}
                role="listbox"
              >
                {models.map((model) => (
                  <div
                    key={model.id}
                    className={cn(
                      "relative cursor-pointer select-none py-1.5 px-2 text-sm rounded-sm",
                      "hover:bg-gray-100 dark:hover:bg-gray-700",
                      selectedModelId === model.id && "bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400"
                    )}
                    onClick={() => handleModelSelect(model)}
                    role="option"
                    aria-selected={selectedModelId === model.id}
                  >
                    <div className="truncate max-w-full">
                      <span>{model.name}</span>
                      <span className="block text-xs text-gray-500 dark:text-gray-400 truncate">
                        {model.description}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default ModelSelector;