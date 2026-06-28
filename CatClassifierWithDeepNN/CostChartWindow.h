//
//  CostChartWindow.h
//  CatClassifierWithDeepNN
//
//  Created by Sugu Lee on 6/28/26.
//

#import <Cocoa/Cocoa.h>

NS_ASSUME_NONNULL_BEGIN

@interface CostChartWindow : NSObject

+ (instancetype)createChartWithTitle:(NSString*)title maxIterations:(int)maxIterations;

- (void)updateWithIteration:(int)iteration cost:(float)cost;

- (void)show;

@end

NS_ASSUME_NONNULL_END
