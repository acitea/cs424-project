import React, { HTMLAttributes } from 'react';
import { cn } from '../utils';
// import { Progress } from '@/components/ui/progress';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface LoadingIndicatorProps extends HTMLAttributes<HTMLDivElement> {
  progress?: number;
  message?: string;
}

interface CustomProgressProps {
  value: number;
  className?: string;
  color?: string;
}

const CustomProgress: React.FC<CustomProgressProps> = ({ 
  value = 0, 
  className = '', 
  color = ''
}) => {
  const safeValue = Math.min(Math.max(value, 0), 100);
  
  return (
    <div className={`w-full rounded-full h-2 ${className}`}>
      <div 
        className={`h-full rounded-full bg-red-200 transition-all duration-300 ease-in-out`}
        style={{ width: `${safeValue}%` }}
      />
    </div>
  );
};

const LoadingIndicator: React.FC<LoadingIndicatorProps> = ({ 
  progress = 0, 
  message = 'Processing...', 
  color = 'bg-red-200',
  className, 
  ...props 
}) => {
  return (
    <Card className={cn('w-full', className)} {...props}>
      <CardContent className="pt-6">
        <div className="flex mb-2 items-center justify-between">
          <div>
            <Badge variant="outline" className="bg-primary/10 text-primary">
              {message}
            </Badge>
          </div>
          <div className="text-right">
            <span className="text-xs font-semibold text-primary">
              {Math.round(progress)}%
            </span>
          </div>
        </div>
        <CustomProgress value={progress} color={color} />
      </CardContent>
    </Card>
  );
};

export default LoadingIndicator;