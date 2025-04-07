import React, { HTMLAttributes } from 'react';
import { cn } from '../utils';
import { Progress } from '@/components/ui/progress';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface LoadingIndicatorProps extends HTMLAttributes<HTMLDivElement> {
  progress?: number;
  message?: string;
}

const LoadingIndicator: React.FC<LoadingIndicatorProps> = ({ 
  progress = 0, 
  message = 'Processing...', 
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
        <Progress value={progress} className="h-2" />
      </CardContent>
    </Card>
  );
};

export default LoadingIndicator;