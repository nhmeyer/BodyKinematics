function [SSPositionWristsmoothed] = StopVelocity(PositionWristsmoothed,StartStop,Failed_trial,a)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
for i = 1:size(PositionWristsmoothed,3)
        if ~ismember(i,Failed_trial(a,2:end))
if StartStop(i,1) == 0 || isnan(StartStop(i,1))
   
    StartStop(i,1) = 1;
    StartStop(i,2)=1;
end

    SSPositionWristsmoothed(1:(StartStop(i,2)-StartStop(i,1))+1,:,i) = PositionWristsmoothed(StartStop(i,1):StartStop(i,2),:,i); % the wrist trajectory end when the wrist velocity end (this is done in order to not take into account the trajectory of the wrist when it goes back to the start point, since some participants forget to stay at the reaching point for one moment)
    SSPositionWristsmoothed(StartStop(i,2)-StartStop(i,1)+2:length(PositionWristsmoothed),:,i) = NaN;
        end
end
end

