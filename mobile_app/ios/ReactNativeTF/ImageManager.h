//
//  ImageManager.h
//  ReactNativeTorch
//
//  Created by ThomasOlivier on 08/06/2016.
//  Copyright © 2016 Facebook. All rights reserved.
//

#ifndef ImageManager_h
#define ImageManager_h


#endif /* ImageManager_h */

#import <UIKit/UIKit.h>
#import "RCTBridgeModule.h"

@interface ImageManager : NSObject <RCTBridgeModule>

+ (CGFloat *)getRGBAsFromImage:(UIImage*)image;
+ (UIImage *)getImageFromRGBA:(CGFloat*)imgTensor channels:(int)channels width:(int)width height:(int)height;

@end