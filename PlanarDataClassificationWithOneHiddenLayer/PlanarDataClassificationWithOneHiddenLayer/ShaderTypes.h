//
//  ShaderTypes.h
//  PlanarDataClassificationWithOneHiddenLayer
//
//  Created by Sugu Lee on 6/20/26.
//

#ifndef ShaderTypes_h
#define ShaderTypes_h

typedef struct
{
    uint32_t numImages;
    uint32_t numFeatures;
    float normalizer_scaler;
} Uniforms;

#endif /* ShaderTypes_h */
