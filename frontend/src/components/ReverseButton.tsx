import React from 'react';
import { ArrowLeftRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/utils';

interface ReverseButtonProps {
  onClick: () => void;
  disabled?: boolean;
  className?: string;
}

const ReverseButton: React.FC<ReverseButtonProps> = ({ 
  onClick, 
  disabled = false,
  className
}) => {
  return (
    <div className={cn("flex flex-col items-center justify-center h-full", className)}>
      <Button
        onClick={onClick}
        disabled={disabled}
        variant="outline"
        size="icon"
        className="rounded-full p-3 h-14 w-14 border-2 hover:bg-white hover:text-black transition-all"
        title="Reverse Input/Output"
      >
        <ArrowLeftRight className="h-8 w-8" />
      </Button>
    </div>
  );
};

export default ReverseButton;