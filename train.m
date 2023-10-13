% 导入训练集
dataFolder1 = 'Hand_Posture_Easy_Stu'; 
gestureLabels = {'A', 'C', 'five', 'V'}; 
numLabels = length(gestureLabels); 
gestureData = imageDatastore(dataFolder1, 'IncludeSubfolders', true, 'LabelSource', 'foldernames'); 
% 读取图像并将其转换为64x64的灰度图像
gestureData.ReadFcn = @(filename)imresize(rgb2gray(imread(filename)), [64 64]); 
trainData=gestureData;

% 计算训练集的HOG特征
cellSize = [8 8]; 
trainFeatures = zeros(numel(trainData.Files), 1764); % 初始化训练集HOG特征矩阵
for i = 1:numel(trainData.Files)
    img = read(trainData);
    trainFeatures(i,:) = extractHOGFeatures(img, 'CellSize', cellSize); % 计算训练集图像的HOG特征
end

% 训练SVM分类器
trainLabels = trainData.Labels; % 获取训练集手势标签
svm = fitcecoc(trainFeatures, trainLabels); 
save('model_save.mat','svm');%将模型保存到model_save.mat
