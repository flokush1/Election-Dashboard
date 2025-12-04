import React from 'react';
import { cn } from '../../shared/utils.js';

export const Card = ({ className, children, ...props }) => (
  <div
    className={cn(
      "rounded-lg border bg-white shadow-sm transition-all hover:shadow-md",
      className
    )}
    {...props}
  >
    {children}
  </div>
);

export const CardHeader = ({ className, children, ...props }) => (
  <div className={cn("flex flex-col space-y-1.5 p-6", className)} {...props}>
    {children}
  </div>
);

export const CardTitle = ({ className, children, ...props }) => (
  <h3
    className={cn(
      "text-2xl font-semibold leading-none tracking-tight",
      className
    )}
    {...props}
  >
    {children}
  </h3>
);

export const CardDescription = ({ className, children, ...props }) => (
  <p
    className={cn("text-sm text-gray-600", className)}
    {...props}
  >
    {children}
  </p>
);

export const CardContent = ({ className, children, ...props }) => (
  <div className={cn("p-6 pt-0", className)} {...props}>
    {children}
  </div>
);

export const CardFooter = ({ className, children, ...props }) => (
  <div className={cn("flex items-center p-6 pt-0", className)} {...props}>
    {children}
  </div>
);

export const Button = ({ className, variant = "default", size = "default", children, ...props }) => {
  const variants = {
    default: "bg-primary text-white hover:bg-primary/90",
    secondary: "bg-gray-100 text-gray-900 hover:bg-gray-200",
    outline: "border border-gray-300 bg-white hover:bg-gray-50",
    ghost: "hover:bg-gray-100",
  };

  const sizes = {
    default: "h-10 px-4 py-2",
    sm: "h-9 px-3",
    lg: "h-11 px-8",
    icon: "h-10 w-10",
  };

  return (
    <button
      className={cn(
        "inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 disabled:opacity-50 disabled:pointer-events-none",
        variants[variant],
        sizes[size],
        className
      )}
      {...props}
    >
      {children}
    </button>
  );
};

export const Badge = ({ className, variant = "default", children, ...props }) => {
  const variants = {
    default: "bg-primary text-white",
    secondary: "bg-gray-100 text-gray-900",
    destructive: "bg-red-500 text-white",
    outline: "border border-gray-300 text-gray-900",
  };

  return (
    <div
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold transition-colors",
        variants[variant],
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};

export const Tabs = ({ value, onValueChange, children, className, ...props }) => (
  <div className={cn("w-full", className)} {...props}>
    {React.Children.map(children, child =>
      React.cloneElement(child, { value, onValueChange })
    )}
  </div>
);

export const TabsList = ({ children, className, ...props }) => (
  <div
    className={cn(
      "inline-flex h-10 items-center justify-center rounded-md bg-gray-100 p-1 text-gray-600",
      className
    )}
    {...props}
  >
    {children}
  </div>
);

export const TabsTrigger = ({ value: triggerValue, onValueChange, value, children, className, ...props }) => (
  <button
    className={cn(
      "inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium ring-offset-white transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
      value === triggerValue
        ? "bg-white text-gray-900 shadow-sm"
        : "text-gray-600 hover:text-gray-900",
      className
    )}
    onClick={() => onValueChange?.(triggerValue)}
    {...props}
  >
    {children}
  </button>
);

export const TabsContent = ({ value: contentValue, value, children, className, ...props }) => {
  if (value !== contentValue) return null;
  
  return (
    <div
      className={cn(
        "mt-2 ring-offset-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};