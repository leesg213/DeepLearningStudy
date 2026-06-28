//
//  DataPlotView.h
//  GoalKeeperDecision
//
//  Created by Sugu Lee on 6/28/26.
//

#import <Cocoa/Cocoa.h>
#include <vector>

NS_ASSUME_NONNULL_BEGIN

@interface DataPlotView : NSView

- (void)setDataWithX:(const std::vector<float>&)x labels:(const std::vector<uint8_t>&)labels;

@end

NS_ASSUME_NONNULL_END
