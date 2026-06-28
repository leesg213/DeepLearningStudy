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
    [self.view addSubview:plotView];
    
    // Pass the training data to the plot view
    [plotView setDataWithX:trainset_x labels:trainset_y];
}


- (void)setRepresentedObject:(id)representedObject {
    [super setRepresentedObject:representedObject];

    // Update the view, if already loaded.
}


@end
