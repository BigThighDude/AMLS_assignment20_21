clear variables
clc

%% Part 1
no_models = 9;  % must be odd
no_out = 2^no_models;
mp = ceil(no_models/2);
ma = 0.838;
mw = 1-ma;
%0.85x9 = 99.44
%0.85x7 = 98.79


x = [0:no_out-1];
y = dec2bin(x);
z = zeros(no_out,1);

for i = 1:no_out
    for m = 1:no_models
        z(i) = z(i)+str2double(y(i,m));
    end
end


tally = zeros(no_models+1, 1);

for i = 1:no_out
    temp = z(i)+1;
    tally(temp) = tally(temp)+1;
end

tot_prob = 0;

for i = 1:mp    % generate probabilities
    exp1 = no_models-i+1;
    exp2 = i-1;
    bin = (ma^exp1)*(mw^exp2);
    tot_prob = (bin*tally(i))+tot_prob;
end

tot_prob*100

% 3 models 0.8 gives 0.8960
% 5 models 0.8 gives 0.9421
% 7 models 0.8 gives 0.9667
% 9 models 0.8 gives 0.9804
% 11 models 0.8 gives 0.9883


%% Part 2
% clear variables
% 
% no_models = 9;  % must be odd
% no_out = 2^no_models;
% mp = ceil(no_models/2);
% ma = 0.8;
% mw = 1-ma;
% 
% x = [0:no_out-1];
% y = dec2bin(x);
% z2 = zeros(no_out,1);
% 
% for i = 1:no_models
%     z2 = z2+str2num(y(:,i));
% end














