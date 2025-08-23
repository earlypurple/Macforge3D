#import <Cocoa/Cocoa.h>

@interface PyQtViewBridge : NSObject

+ (NSView *)createViewWithPyQtWidget:(void *)widget;
+ (void)destroyView:(NSView *)view;

@end
