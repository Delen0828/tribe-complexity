function DataVizThesis_HAScript(subject, listName, runs)

% MRI script for the HAshcraft DataViz Thesis
% subject = subject ID# | subject =  %fill the subjectID# | listName = stimulus list .xlsx name | runs = number of runs
try

% initial parameters
blockFixationTime = 12;
intertrialFixationTime = 0.5;
promptTime = 3;
stim_t = 3; % stimulus duration in seconds
figsPerBlock = 6;
block1 = "XPQMOPMXQOPQMOX";
block2 = reverse(block1);

%%
% input checks

if nargin < 1 || isempty(subject)
    error("Provide subject");
end

if nargin < 2 || isempty(listName)
    error("Provide stimulus list");
end

if nargin < 3 || isempty(runs)
    error("Provide number of runs");
end

listName = char(strtrim(string(listName)));
subjID = char(string(subject));
[~, stimBase, listExt] = fileparts(listName);

if isempty(listExt)
    listExt = '.xlsx';
end

% stimulus folder naming
stimFolderName = strrep(stimBase, 'stimSet', 'dataVizSet');
fontColor = [0 0 0];
nBlocksPerRun = sum(char(block1) ~= 'X');
nFigsPerRun = nBlocksPerRun * figsPerBlock;

%%
% random number generator setup
rng(1234);
seed = 1234;

%%
% run_list (should be .xlsx)
listFile = fullfile(pwd, [stimBase listExt]);
T = readtable(listFile, 'VariableNamingRule', 'preserve');
stimNames = strtrim(string(T{:,1}));
nFigs = height(T);

if nFigs ~= nFigsPerRun || length(unique(stimNames)) ~= nFigsPerRun
    error("unique stimulus filenames are required");
end

%%
% correlation and outlier values are contained in stimulus filename
rValue = nan(nFigs, 1);
isPositive = false(nFigs, 1);
isNegative = false(nFigs, 1);
isOutlier = false(nFigs, 1);
outlierMetric = nan(nFigs, 1);

for i = 1:nFigs
    rToken = regexp(char(stimNames(i)),'\(r=([+-]?(?:\d+(?:\.\d*)?|\.\d+))\)', 'tokens', 'once');

    if ~isempty(rToken)
        rValue(i) = str2double(rToken{1});
    end
    isPositive(i) = contains(stimNames(i), "_pos");
    isNegative(i) = contains(stimNames(i), "_neg");
    isOutlier(i) = contains(stimNames(i), "_out");

    if isOutlier(i)
        zToken = regexp(char(stimNames(i)), ',z=([+-]?(?:\d+(?:\.\d*)?|\.\d+))\)', 'tokens', 'once');

        if ~isempty(zToken)
            outlierMetric(i) = str2double(zToken{1});
        end
    end
end

%%
% stimulus condition pools
posOutPool = find(isPositive & isOutlier);
posNoOutPool = find(isPositive & ~isOutlier);
negOutPool = find(isNegative & isOutlier);
negNoOutPool = find(isNegative & ~isOutlier);

if any(isPositive == isNegative) || any(isnan(rValue))
    error("Each stimulus must have a valid positive or negative correlation");
end

if length(posOutPool) ~= length(negNoOutPool) || length(posNoOutPool) ~= length(negOutPool) || length(posOutPool) + length(posNoOutPool) ~= nFigsPerRun / 2 || length(posOutPool) + length(negOutPool) ~= nFigsPerRun / 2
    error("Stimulus pool cannot support balanced task assignments");
end

%%
% output file naming; prevents overwriting prior runs
runNum = 1;

while true
    runFilesExist = false;

    for r = 0:runs-1

        if exist(sprintf('%s_run%d.csv', subjID, runNum + r), 'file')
            runFilesExist = true;

            break

        end
    end

    if ~runFilesExist
        break
    end

    runNum = runNum + 1;
end

%%
% PTB setup
KbName('UnifyKeyNames');
PsychDefaultSetup(2);
Screen('Preference', 'SkipSyncTests', 1);
scr = max(Screen('Screens'));
white = WhiteIndex(scr);
[win, rect] = PsychImaging('OpenWindow', scr, white);
[sw, sh] = Screen('WindowSize', win);
[cx, cy] = RectCenter(rect);
Screen('TextSize', win, 80);
Screen('TextFont', win, 'Times New Roman');
questionWrap = max(20, floor(sw / 40));

%%
% image preload
stimDir = fullfile(pwd, stimFolderName);
usedNames = unique(stimNames);
texMap = containers.Map;

for i = 1:length(usedNames)
    name = char(usedNames(i));
    imgPath = fullfile(stimDir, name);
    texMap(name) = Screen('MakeTexture', win, imread(imgPath));
end

%%
% fMRI trigger setup
addpath([pwd filesep 'TOOL_Matlab_trigger_utils']);

%%
% run loop (not the experimental loop)
for run = 1:runs

    %%
    % output file naming
    currentRunNum = runNum + run - 1;
    outFile = sprintf('%s_run%d.csv', subjID, currentRunNum);

    %%
    % stimulus condition pools
    if run == 1
        posOutPool = posOutPool(randperm(length(posOutPool)));
        posNoOutPool = posNoOutPool(randperm(length(posNoOutPool)));
        negOutPool = negOutPool(randperm(length(negOutPool)));
        negNoOutPool = negNoOutPool(randperm(length(negNoOutPool)));
    end

    %%
    % preassign balanced stimuli to each task
    taskCodes = unique(char(erase(block1, "X")), 'stable');
    taskRows = cell(1, length(taskCodes));

    if run == 1
        remainingPosOut = length(posOutPool);
        posOutCount = 0;
        posNoOutCount = 0;
        negOutCount = 0;
        negNoOutCount = 0;

        for t = 1:length(taskCodes)
            remainingTasks = length(taskCodes) - t;
            minPosOut = max(0, remainingPosOut - 9 * remainingTasks);
            maxPosOut = min(9, remainingPosOut);
            nPosOut = randi([minPosOut maxPosOut]);
            nPosNoOut = 9 - nPosOut;
            nNegOut = 9 - nPosOut;
            nNegNoOut = nPosOut;
            posOutRows = posOutPool(posOutCount + (1:nPosOut));
            posNoOutRows = posNoOutPool(posNoOutCount + (1:nPosNoOut));
            negOutRows = negOutPool(negOutCount + (1:nNegOut));
            negNoOutRows = negNoOutPool(negNoOutCount + (1:nNegNoOut));
            posOutCount = posOutCount + nPosOut;
            posNoOutCount = posNoOutCount + nPosNoOut;
            negOutCount = negOutCount + nNegOut;
            negNoOutCount = negNoOutCount + nNegNoOut;
            remainingPosOut = remainingPosOut - nPosOut;
            assignedRows = [posOutRows(:); posNoOutRows(:); negOutRows(:); negNoOutRows(:)];
            taskRows{t} = assignedRows(randperm(length(assignedRows)));
        end

        firstRunTaskRows = taskRows;

    else
        taskRows = firstRunTaskRows([end, 1:end-1]);
    end

    %%
    % hardcoded block order
    if run == 1
        blockMask = char(block1);
    else
        blockMask = char(block2);
    end
    runOrder = string(blockMask);

    %%
    % motion assignment; 0 = Left, 1 = Right no more than 3 consecutive same directions
    motionVec = zeros(nFigsPerRun, 1);

    for i = 1:nFigsPerRun
      
        if i >= 4 && all(motionVec(i-3:i-1) == motionVec(i-1))
            motionVec(i) = 1 - motionVec(i-1);
        else
            motionVec(i) = randi([0 1]);
        end
    end

    %%
    % fMRI trigger setup
    Screen('FillRect', win, white);
    DrawFormattedText(win, 'Waiting for scanner...', 'center', 'center', fontColor);
    Screen('Flip', win);
    run_start_time = wait_for_trigger_kbqueue_all_dvc();

    %%
    % output setup
    fid = fopen(outFile, 'w');

    if fid == -1
        error('Could not open output file: %s', outFile);
    end

    fprintf(fid, 'Seed = %u\n', seed);
    fprintf(fid, 'runOrder = %s\n', char(runOrder));
    fprintf(fid, 'Trial,Block,TrialCode,Stimulus,r,z,Question,Direction,Response,CorrectAns,RT,Correct\n');
    movementCorrect = [];
    questionCorrect = [];
    outlierCorrect = [];
    trialNum = 0;

    %%
    % block loop
    blockNum = 0;

    for b = 1:length(blockMask)
        trialCode = blockMask(b);

        if trialCode == 'X'
            Screen('FillRect', win, white);
            DrawFormattedText(win, '+', 'center', 'center', fontColor);
            Screen('Flip', win);
            fixStart = GetSecs;

            while GetSecs - fixStart < blockFixationTime
                [kd, ~, kc] = KbCheck;

                if kd && kc(KbName('ESCAPE'))
                    error('escaped loop');
                end
            end

            continue

        end
        blockNum = blockNum + 1;

        %%
        % task prompt
        if trialCode == 'M'
            questionText = "Which direction is the figure moving?\n1 = left, 2 = right";
        elseif trialCode == 'Q'
            questionText = "What is trend in the data?\n1 = positive trend, 2 = negative trend"; %tell them to look at the dots
        elseif trialCode == 'O'
            questionText = "Is there an outlier in the data?\n1 = yes, 2 = no";
        else % P
            questionText = "Press 1 when you see the figure.";
        end

        Screen('FillRect', win, white);
        DrawFormattedText(win, char(questionText), 'center', 'center', fontColor, questionWrap);
        Screen('Flip', win);
        promptStart = GetSecs;

        while GetSecs - promptStart < promptTime
            [kd, ~, kc] = KbCheck;

            if kd && kc(KbName('ESCAPE'))
                error('escaped loop');
            end
        end

        %%
        % select 6 randomized unused figures from the preassigned task pool
        taskIndex = find(taskCodes == trialCode, 1);
        availableRows = taskRows{taskIndex};
        selectedIndices = randperm(length(availableRows), figsPerBlock);
        blockRows = availableRows(selectedIndices);
        taskRows{taskIndex}(selectedIndices) = [];

        %%
        % figures within block
        for s = 1:figsPerBlock
            trialNum = trialNum + 1;
            stimRow = blockRows(s);
            stimName = stimNames(stimRow);

            %%
            % stimulus
            texStim = texMap(char(stimName));
            srcRect = Screen('Rect', texStim);
            srcW = srcRect(3) - srcRect(1);
            srcH = srcRect(4) - srcRect(2);

            %%
            % movement
            motionCode = motionVec(trialNum);
            directionSign = 2 * motionCode - 1;

            if motionCode == 0
                directionText = "Left (1)";
            else
                directionText = "Right (2)";
            end

            %%
            % correct answer
            if trialCode == 'M'
                correctAnswer = motionCode + 1;
            elseif trialCode == 'Q'
                correctAnswer = 1 + isNegative(stimRow);
            elseif trialCode == 'O'
                correctAnswer = 1 + ~isOutlier(stimRow); % 1 = Yes, 2 = No
            else % P
                correctAnswer = NaN;
            end

            %%
            % stimulus presentation
            imgScale = min((sw * .80) / srcW, (sh * .80) / srcH);
            baseRect = [0 0 round(srcW * imgScale) round(srcH * imgScale)];
            totalTravelPixels = sw * .005; % movement speed
            startX = cx - directionSign * totalTravelPixels / 2;
            endX = cx + directionSign * totalTravelPixels / 2;
            pressed = false;
            respKey = NaN;
            rt = NaN;
            firstFlip = NaN;

            %%
            % stimulus + fixation loop
            while isnan(firstFlip) || GetSecs - firstFlip < stim_t + intertrialFixationTime

                if isnan(firstFlip)
                    elapsed = 0;
                else
                    elapsed = GetSecs - firstFlip;
                end

                Screen('FillRect', win, white);

                if elapsed < stim_t
                    proportion = min(elapsed / stim_t, 1);
                    stimX = startX + proportion * (endX - startX);
                    dstRect = CenterRectOnPointd(baseRect, stimX, cy);
                    Screen('DrawTexture', win, texStim, [], dstRect, [], 1);
                else
                    DrawFormattedText(win, '+', 'center', 'center', fontColor);
                end

                flipTime = Screen('Flip', win);

                if isnan(firstFlip)
                    firstFlip = flipTime;
                end

                [kd, ~, kc] = KbCheck;

                if kd

                    if kc(KbName('ESCAPE'))
                        error('escaped loop');
                    end

                    if ~pressed

                        if kc(KbName('1!')) || kc(KbName('1'))
                            respKey = 1;
                        elseif trialCode ~= 'P' && (kc(KbName('2@')) || kc(KbName('2')))
                            respKey = 2;
                        end

                        if ~isnan(respKey)
                            pressed = true;
                            rt = GetSecs - firstFlip;
                        end
                    end
                end
            end

            %%
            % save response

            if trialCode == 'P'
                correct = NaN;
            else
                correct = double(~isnan(respKey) && respKey == correctAnswer);

                if trialCode == 'M'
                    movementCorrect(end + 1) = correct;
                elseif trialCode == 'Q'
                    questionCorrect(end + 1) = correct;
                elseif trialCode == 'O'
                    outlierCorrect(end + 1) = correct;
                end
            end

           if trialCode == 'P' || isnan(respKey)
                respStr = "";
            else
                respStr = sprintf('%.0f', respKey);
           end

            if isnan(correctAnswer)
                correctAnswerStr = "";
            else
                correctAnswerStr = sprintf('%.0f', correctAnswer);
            end

            if isnan(rt)
                rtStr = "";
            else
                rtStr = sprintf('%.6f', rt);
            end

            if isnan(correct)
                correctStr = "";
            else
                correctStr = sprintf('%.0f', correct);
            end

            if isnan(outlierMetric(stimRow))
                outlierMetricStr = "";
            else
                outlierMetricStr = sprintf('%.6f', outlierMetric(stimRow));
            end

            if trialCode == 'M'
                trialCodeOut = "Movement (M)";
            elseif trialCode == 'Q'
                trialCodeOut = "Question (Q)";
            elseif trialCode == 'O'
                trialCodeOut = "Outlier (O)";
            else % P
                trialCodeOut = "Passive (P)";
            end

            stimOut = strrep(char(stimName), '"', '""');
            questionOut = strrep(char(questionText), '"', '""');
            fprintf(fid,'%d,%d,"%s","%s",%.6f,%s,"%s","%s",%s,%s,%s,%s\n', trialNum, blockNum, trialCodeOut, stimOut, rValue(stimRow), outlierMetricStr, questionOut, directionText, respStr, correctAnswerStr, rtStr, correctStr);
        end
    end

    %%
    % data collection/accuracy
    movementAcc = 100 * mean(movementCorrect, 'omitnan');
    questionAcc = 100 * mean(questionCorrect, 'omitnan');
    outlierAcc = 100 * mean(outlierCorrect, 'omitnan');
    fprintf(fid, '\nMovement Accuracy,%.1f%%\n', movementAcc);
    fprintf(fid, '\nQuestion Accuracy,%.1f%%\n', questionAcc);
    fprintf(fid, '\nOutlier Accuracy,%.1f%%\n', outlierAcc);
    fclose(fid);
    fid = -1;
end

%%
% close PTB
texKeys = keys(texMap);

for i = 1:length(texKeys)
    Screen('Close', texMap(texKeys{i}));
end

sca;

catch ME

sca;
ListenChar(0);

if exist('fid', 'var') && fid > 0

    try
        fclose(fid);
    catch

    end
end

rethrow(ME);
end
end