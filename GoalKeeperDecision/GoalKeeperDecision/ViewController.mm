//
//  ViewController.m
//  GoalKeeperDecision
//
//  Created by Sugu Lee on 6/28/26.
//

#import "ViewController.h"
#import "DataPlotView.h"
#include <vector>
#include <H5Cpp.h>
#include <matio.h>
#include "DeepHiddenLayerModel.hpp"

@implementation ViewController
{
    std::vector<float> trainset_x, testset_x;
    std::vector<uint8_t> trainset_y, testset_y;
    DataPlotView* plotView;
}
-(std::vector<float>)load_dataset_x:(matvar_t*)matvar
{
    std::vector<float> output;
    
    size_t cols = matvar->dims[1];
    size_t rows = matvar->dims[0];
    
    // Check if the data type matches what you expect (e.g., Double Precision)
    if (matvar->class_type == MAT_C_DOUBLE && matvar->data_type == MAT_T_DOUBLE) {
        // Cast the internal data pointer safely
        double* data_ptr = static_cast<double*>(matvar->data);

        for (size_t r = 0; r < rows; ++r)
        {
            for (size_t c = 0; c < cols; ++c) {
                // Formula to map 2D coordinates to column-major array
                size_t index = c * rows + r;
                double data_entry = data_ptr[index];
                
                output.push_back((float)data_entry);
            }
        }
    } else {
        
    }
    
    return output;
}
-(std::vector<uint8_t>)load_dataset_y:(matvar_t*)matvar
{
    std::vector<uint8_t> output;
    
    size_t cols = matvar->dims[1];
    size_t rows = matvar->dims[0];
    
    // Check if the data type matches what you expect (e.g., Double Precision)
    if (matvar->class_type == MAT_C_DOUBLE && matvar->data_type == MAT_T_DOUBLE) {
        // Cast the internal data pointer safely
        double* data_ptr = static_cast<double*>(matvar->data);

        for (size_t c = 0; c < cols; ++c) {
            for (size_t r = 0; r < rows; ++r) {
                // Formula to map 2D coordinates to column-major array
                size_t index = c * rows + r;
                double data_entry = data_ptr[index];
                
                output.push_back((float)data_entry);
            }
        }
    } else {
        
    }
    
    return output;
}
- (void)loadDataSets
{
    NSString* dataset_path = @"/Users/sugulee/Documents/GitHub/DeepLearningStudy/Files/home/jovyan/work/release/W1A2/datasets/data.mat";
    
    mat_t* matfp = Mat_Open(dataset_path.UTF8String, MAT_ACC_RDONLY);
    
    matvar_t* train_set_x = Mat_VarRead(matfp, "X");
    matvar_t* train_set_y = Mat_VarRead(matfp, "y");
    matvar_t* test_set_x = Mat_VarRead(matfp, "Xval");
    matvar_t* test_set_y = Mat_VarRead(matfp, "yval");
    
    trainset_x = [self load_dataset_x:train_set_x];
    trainset_y = [self load_dataset_y:train_set_y];
    testset_x = [self load_dataset_x:test_set_x];
    testset_y = [self load_dataset_y:test_set_y];
    
    Mat_VarFree(train_set_x);
    Mat_VarFree(train_set_y);
    Mat_VarFree(test_set_x);
    Mat_VarFree(test_set_y);
    
    Mat_Close(matfp);
    
    NSLog(@"Datasets loaded. Num train : %d Num test : %d", trainset_y.size(), testset_y.size());
}
- (void)viewDidLoad {
    [super viewDidLoad];

    // Do any additional setup after loading the view.
    
    [self loadDataSets];
    
    // Create and add the plot view
    NSRect frame = self.view.bounds;
    plotView = [[DataPlotView alloc] initWithFrame:frame];
    plotView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    [self.view addSubview:plotView positioned:NSWindowBelow relativeTo:nil];
    
    // Pass the training data to the plot view
    [plotView setDataWithX:trainset_x labels:trainset_y];
}

- (IBAction)runDeepHiddenLayerModel:(id)sender
{
    // Run training on background thread to keep UI responsive
   // dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        DeepHiddenLayerModel _DeepHiddenLayerModel;
        
        std::vector layers = {0, 20, 3, 1};
        layers[0] = 2;
        _DeepHiddenLayerModel.Init(layers);
        
        // Train with real-time cost updates
        _DeepHiddenLayerModel.TrainF(trainset_x, trainset_y, 30000, 0.3f, 0.7f, 100, nullptr,
            [](int iteration, float cost) {
               // [chartWindow updateWithIteration:iteration cost:cost];
            });
        
        // Plot decision boundary
        {
            std::vector<float> dataset_x;
            std::vector<float> predict;
            float x_min = -0.8f;
            float x_max = 0.5f;
            float y_min = -0.75f;
            float y_max = 0.75f;
            int num_points = 100;
            float interval_x = (x_max-x_min) / num_points;
            float interval_y = (y_max-y_min) / num_points;
            for(float x = x_min;x<x_max; x+= interval_x)
            {
                for(float y = y_min;y<y_max; y+= interval_y)
                {
                    dataset_x.push_back(x);
                    dataset_x.push_back(y);
                }
            }
            
            _DeepHiddenLayerModel.PredictF(dataset_x, dataset_x.size()/2, predict);
            
            std::vector<uint8_t> labels;
            for(int i = 0;i<predict.size();++i)
            {
                labels.push_back(predict[i] >0.5f ? 1 : 0);
            }
            
            [plotView plotDecisionBoundary:dataset_x labels:labels];
        }
        
        // Prediction also on background thread
        auto predict = [&](std::vector<float> const& set_x, std::vector<uint8_t>& set_y, std::string const& name)
        {
            size_t num_sets = set_y.size();
            
            std::vector<float> results;
            _DeepHiddenLayerModel.PredictF(set_x, num_sets, results);
            int num_correct = 0;
            for(int i = 0;i<results.size();++i)
            {
                int predict = results[i] >= 0.5f;
                if(predict == set_y[i]) num_correct++;
            }
            NSLog(@"[%s] Predict : %d / %d (%.3f%)", name.c_str(), num_correct, results.size(), num_correct/((float)results.size())*100.0f);
        };
        
        predict(trainset_x, trainset_y, "TrainSet");
        predict(testset_x, testset_y, "TestSet");
   // });
}

- (void)setRepresentedObject:(id)representedObject {
    [super setRepresentedObject:representedObject];

    // Update the view, if already loaded.
}


@end
