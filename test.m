%导入测试集，加载训练好的SVM分类器模型
model = load('model_save.mat');%读取训练好的分类模型
svm = model.svm;
cellSize = [8 8]; % 每个单元格大小

dataFolder2 = 'test'; % 文件夹路径
files = dir(fullfile(dataFolder2, '*.png')); % 获取文件夹中所有PNG格式的文件

testData = imageDatastore(dataFolder2, 'FileExtensions', '.png', 'LabelSource', 'foldernames'); 
testData.ReadFcn = @(filename)imresize(rgb2gray(imread(filename)), [64 64]);

% 提取PNG图片名称的前部分作为标签
for i = 1:numel(testData.Files)
    [~, filename, ~] = fileparts(testData.Files{i});
    label = extractBefore(filename, '-'); 
    label = cellstr(label); 
    testData.Labels(i) = categorical(label); 
end

%计算测试集的HOG特征
testFeatures = zeros(numel(testData.Files), 1764); % 初始化测试集HOG特征矩阵
for i = 1:numel(testData.Files)
    img = readimage(testData, i);
    % 计算测试集图像的HOG特征
    testFeatures(i,:) = extractHOGFeatures(img, 'CellSize', cellSize); 
end

% 测试SVM分类器，对测试集数据进行分类
testLabels = testData.Labels; 
predictedLabels = predict(svm, testFeatures); 

%显示分类结果和分类图像
figure;
for i = 1:numel(testData.Files)
    subplot(10, 4, i);
    img = readimage(testData, i);
    imshow(img);
    trueLabel = char(testLabels(i));
    predictedLabel = char(predictedLabels(i));
    text(-100, 100, [' ' ' 预测为： ' predictedLabel], 'Color', 'b');
end

