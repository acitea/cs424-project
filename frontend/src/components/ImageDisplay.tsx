import React, { HTMLAttributes } from 'react';
import { Download, ImageIcon } from 'lucide-react';
import { cn } from '../utils';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

interface ImageDisplayProps extends HTMLAttributes<HTMLDivElement> {
  title?: string;
  image?: string | null;
}

const ImageDisplay: React.FC<ImageDisplayProps> = ({ 
  title = 'Result', 
  image, 
  className, 
  ...props 
}) => {
  const handleDownload = () => {
    if (!image) return;
    
    const link = document.createElement('a');
    link.href = image;
    link.download = 'transformed-image.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <Card className={cn('w-full', className)} {...props}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      
      <CardContent className="p-4">
        <div className="image-preview bg-muted">
          {image ? (
            <img src={image} alt={title} className="max-w-full max-h-full object-contain" />
          ) : (
            <div className="flex flex-col items-center justify-center text-muted-foreground">
              <ImageIcon className="w-12 h-12 mb-2" />
              <span className="text-sm">No image to display yet</span>
            </div>
          )}
        </div>
      </CardContent>
      
      {image && (
        <CardFooter className="flex justify-end">
          <Button 
            onClick={handleDownload}
            variant="outline"
            className="flex items-center gap-2"
            type="button"
          >
            <Download size={18} />
            <span>Download</span>
          </Button>
        </CardFooter>
      )}
    </Card>
  );
};

export default ImageDisplay;