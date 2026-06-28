//
//  CostChartWindow.m
//  CatClassifierWithDeepNN
//
//  Created by Sugu Lee on 6/28/26.
//

#import "CostChartWindow.h"
#import <Quartz/Quartz.h>

// Forward declarations - moved to top
@interface CostChartView : NSView
@property (nonatomic, strong) NSMutableArray<NSNumber*>* iterations;
@property (nonatomic, strong) NSMutableArray<NSNumber*>* costs;
@property (nonatomic, strong) NSString* chartTitle;
@property (nonatomic, assign) int maxIterations;
- (instancetype)initWithFrame:(NSRect)frameRect title:(NSString*)title maxIterations:(int)maxIterations;
- (void)updateWithIteration:(int)iteration cost:(float)cost;
@end

@interface CostChartWindowController : NSWindowController
@end

@interface CostChartWindow()
@property (nonatomic, strong) NSWindow* window;
@property (nonatomic, strong) CostChartView* chartView;
@property (nonatomic, strong) CostChartWindowController* controller;
@end

@implementation CostChartWindow

+ (instancetype)createChartWithTitle:(NSString*)title maxIterations:(int)maxIterations {
    CostChartWindow* chart = [[CostChartWindow alloc] init];
    
    // Create window
    chart.window = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 800, 600)
                                                styleMask:(NSWindowStyleMaskTitled |
                                                          NSWindowStyleMaskClosable |
                                                          NSWindowStyleMaskMiniaturizable |
                                                          NSWindowStyleMaskResizable)
                                                  backing:NSBackingStoreBuffered
                                                    defer:NO];
    [chart.window setTitle:title];
    [chart.window center];
    
    // Create custom view
    chart.chartView = [[CostChartView alloc] initWithFrame:chart.window.contentView.bounds
                                                     title:title
                                             maxIterations:maxIterations];
    chart.chartView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    
    [chart.window setContentView:chart.chartView];
    
    // Keep window controller alive
    chart.controller = [[CostChartWindowController alloc] initWithWindow:chart.window];
    
    return chart;
}

- (void)show {
    [self.window makeKeyAndOrderFront:nil];
}

- (void)updateWithIteration:(int)iteration cost:(float)cost {
    // Update on main thread
    dispatch_async(dispatch_get_main_queue(), ^{
        [self.chartView updateWithIteration:iteration cost:cost];
    });
}

@end

@implementation CostChartWindowController
@end

// CostChartView implementation

@implementation CostChartView

- (instancetype)initWithFrame:(NSRect)frameRect title:(NSString*)title maxIterations:(int)maxIterations {
    self = [super initWithFrame:frameRect];
    if (self) {
        _iterations = [NSMutableArray array];
        _costs = [NSMutableArray array];
        _chartTitle = title;
        _maxIterations = maxIterations;
    }
    return self;
}

- (void)updateWithIteration:(int)iteration cost:(float)cost {
    [self.iterations addObject:@(iteration)];
    [self.costs addObject:@(cost)];
    [self setNeedsDisplay:YES];
}

- (void)drawRect:(NSRect)dirtyRect {
    [super drawRect:dirtyRect];
    
    if (self.iterations.count == 0 || self.costs.count == 0) {
        NSString* noDataText = @"No data to display";
        NSDictionary* attributes = @{
            NSFontAttributeName: [NSFont systemFontOfSize:16],
            NSForegroundColorAttributeName: [NSColor secondaryLabelColor]
        };
        NSSize textSize = [noDataText sizeWithAttributes:attributes];
        NSPoint textPoint = NSMakePoint((self.bounds.size.width - textSize.width) / 2,
                                       (self.bounds.size.height - textSize.height) / 2);
        [noDataText drawAtPoint:textPoint withAttributes:attributes];
        return;
    }
    
    // Draw background
    [[NSColor windowBackgroundColor] setFill];
    NSRectFill(dirtyRect);
    
    // Margins
    CGFloat leftMargin = 80;
    CGFloat rightMargin = 40;
    CGFloat topMargin = 80;
    CGFloat bottomMargin = 100;
    CGFloat statsHeight = 150;
    
    // Chart area
    CGRect chartRect = NSMakeRect(leftMargin,
                                 bottomMargin + statsHeight,
                                 self.bounds.size.width - leftMargin - rightMargin,
                                 self.bounds.size.height - topMargin - bottomMargin - statsHeight);
    
    // Set fixed axis ranges
    int minIter = 0;
    int maxIter = self.maxIterations;
    double minCost = 0.0;
    double maxCost = 1.0;
    
    // Draw title
    NSDictionary* titleAttrs = @{
        NSFontAttributeName: [NSFont boldSystemFontOfSize:20],
        NSForegroundColorAttributeName: [NSColor labelColor]
    };
    [self.chartTitle drawAtPoint:NSMakePoint(leftMargin, self.bounds.size.height - topMargin + 40)
                  withAttributes:titleAttrs];
    
    // Draw axes
    [[NSColor gridColor] setStroke];
    NSBezierPath* axisPath = [NSBezierPath bezierPath];
    [axisPath moveToPoint:NSMakePoint(chartRect.origin.x, chartRect.origin.y)];
    [axisPath lineToPoint:NSMakePoint(chartRect.origin.x, chartRect.origin.y + chartRect.size.height)];
    [axisPath moveToPoint:NSMakePoint(chartRect.origin.x, chartRect.origin.y)];
    [axisPath lineToPoint:NSMakePoint(chartRect.origin.x + chartRect.size.width, chartRect.origin.y)];
    [axisPath setLineWidth:2.0];
    [axisPath stroke];
    
    // Draw grid lines and y-axis labels
    int numYGridLines = 5;
    NSDictionary* labelAttrs = @{
        NSFontAttributeName: [NSFont systemFontOfSize:11],
        NSForegroundColorAttributeName: [NSColor secondaryLabelColor]
    };
    
    [[NSColor gridColor] setStroke];
    for (int i = 0; i <= numYGridLines; i++) {
        double value = minCost + (maxCost - minCost) * i / numYGridLines;
        CGFloat y = chartRect.origin.y + chartRect.size.height * i / numYGridLines;
        
        // Grid line
        NSBezierPath* gridLine = [NSBezierPath bezierPath];
        [gridLine moveToPoint:NSMakePoint(chartRect.origin.x, y)];
        [gridLine lineToPoint:NSMakePoint(chartRect.origin.x + chartRect.size.width, y)];
        [gridLine setLineWidth:0.5];
        [[NSColor gridColor] setStroke];
        [gridLine stroke];
        
        // Label
        NSString* label = [NSString stringWithFormat:@"%.4f", value];
        NSSize labelSize = [label sizeWithAttributes:labelAttrs];
        [label drawAtPoint:NSMakePoint(chartRect.origin.x - labelSize.width - 10, y - labelSize.height / 2)
            withAttributes:labelAttrs];
    }
    
    // Draw x-axis labels
    int numXLabels = 6;
    for (int i = 0; i <= numXLabels; i++) {
        int value = minIter + (maxIter - minIter) * i / numXLabels;
        CGFloat x = chartRect.origin.x + chartRect.size.width * i / numXLabels;
        
        NSString* label = [NSString stringWithFormat:@"%d", value];
        NSSize labelSize = [label sizeWithAttributes:labelAttrs];
        [label drawAtPoint:NSMakePoint(x - labelSize.width / 2, chartRect.origin.y - 25)
            withAttributes:labelAttrs];
    }
    
    // Draw axis labels
    NSDictionary* axisLabelAttrs = @{
        NSFontAttributeName: [NSFont systemFontOfSize:13],
        NSForegroundColorAttributeName: [NSColor labelColor]
    };
    
    NSString* xAxisLabel = @"Iteration";
    NSSize xLabelSize = [xAxisLabel sizeWithAttributes:axisLabelAttrs];
    [xAxisLabel drawAtPoint:NSMakePoint(chartRect.origin.x + (chartRect.size.width - xLabelSize.width) / 2,
                                       chartRect.origin.y - 50)
             withAttributes:axisLabelAttrs];
    
    // Draw data line
    if (self.costs.count > 1) {
        NSBezierPath* linePath = [NSBezierPath bezierPath];
        
        for (NSInteger i = 0; i < self.costs.count; i++) {
            int iter = [self.iterations[i] intValue];
            double cost = [self.costs[i] doubleValue];
            
            CGFloat x = chartRect.origin.x + chartRect.size.width * (iter - minIter) / (maxIter - minIter);
            CGFloat y = chartRect.origin.y + chartRect.size.height * (cost - minCost) / (maxCost - minCost);
            
            if (i == 0) {
                [linePath moveToPoint:NSMakePoint(x, y)];
            } else {
                [linePath lineToPoint:NSMakePoint(x, y)];
            }
        }
        
        [[NSColor systemBlueColor] setStroke];
        [linePath setLineWidth:2.0];
        [linePath stroke];
        
        // Draw points
        for (NSInteger i = 0; i < self.costs.count; i++) {
            int iter = [self.iterations[i] intValue];
            double cost = [self.costs[i] doubleValue];
            
            CGFloat x = chartRect.origin.x + chartRect.size.width * (iter - minIter) / (maxIter - minIter);
            CGFloat y = chartRect.origin.y + chartRect.size.height * (cost - minCost) / (maxCost - minCost);
            
            NSBezierPath* circle = [NSBezierPath bezierPathWithOvalInRect:NSMakeRect(x - 3, y - 3, 6, 6)];
            [[NSColor systemBlueColor] setFill];
            [circle fill];
        }
    }
    
    // Draw statistics panel
    CGRect statsRect = NSMakeRect(leftMargin, bottomMargin, 
                                 self.bounds.size.width - leftMargin - rightMargin, 
                                 statsHeight - 20);
    
    [[NSColor controlBackgroundColor] setFill];
    NSBezierPath* statsBackground = [NSBezierPath bezierPathWithRoundedRect:statsRect xRadius:8 yRadius:8];
    [statsBackground fill];
    
    [[NSColor separatorColor] setStroke];
    [statsBackground setLineWidth:1.0];
    [statsBackground stroke];
    
    // Calculate statistics from actual data
    double firstCost = [self.costs.firstObject doubleValue];
    double lastCost = [self.costs.lastObject doubleValue];
    double costReduction = ((firstCost - lastCost) / firstCost) * 100.0;
    
    // Find actual min/max for statistics display
    double actualMinCost = firstCost;
    double actualMaxCost = firstCost;
    for (NSNumber* cost in self.costs) {
        double val = [cost doubleValue];
        if (val < actualMinCost) actualMinCost = val;
        if (val > actualMaxCost) actualMaxCost = val;
    }
    
    NSDictionary* statLabelAttrs = @{
        NSFontAttributeName: [NSFont boldSystemFontOfSize:12],
        NSForegroundColorAttributeName: [NSColor labelColor]
    };
    NSDictionary* statValueAttrs = @{
        NSFontAttributeName: [NSFont systemFontOfSize:12],
        NSForegroundColorAttributeName: [NSColor secondaryLabelColor]
    };
    
    __block CGFloat statY = statsRect.origin.y + statsRect.size.height - 30;
    CGFloat statSpacing = 22;
    CGFloat statX = statsRect.origin.x + 20;
    
    // Helper function to draw stat
    void (^drawStat)(NSString*, NSString*, NSColor*) = ^(NSString* label, NSString* value, NSColor* valueColor) {
        [label drawAtPoint:NSMakePoint(statX, statY) withAttributes:statLabelAttrs];
        
        NSMutableDictionary* valueAttrs = [statValueAttrs mutableCopy];
        if (valueColor) {
            valueAttrs[NSForegroundColorAttributeName] = valueColor;
        }
        
        NSSize labelSize = [label sizeWithAttributes:statLabelAttrs];
        [value drawAtPoint:NSMakePoint(statX + labelSize.width + 10, statY) withAttributes:valueAttrs];
        
        statY -= statSpacing;
    };
    
    drawStat(@"Initial Cost:", [NSString stringWithFormat:@"%.4f", firstCost], nil);
    drawStat(@"Final Cost:", [NSString stringWithFormat:@"%.4f", lastCost], nil);
    drawStat(@"Min Cost:", [NSString stringWithFormat:@"%.4f", actualMinCost], [NSColor systemGreenColor]);
    drawStat(@"Max Cost:", [NSString stringWithFormat:@"%.4f", actualMaxCost], [NSColor systemRedColor]);
    drawStat(@"Cost Reduction:", [NSString stringWithFormat:@"%.2f%%", costReduction], 
            costReduction > 0 ? [NSColor systemGreenColor] : [NSColor systemRedColor]);
}

@end
