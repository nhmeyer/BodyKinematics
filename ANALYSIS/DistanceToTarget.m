function [WristDistanceA0,WristDistanceA25,WristDistanceAM25,WristDistanceTot] = DistanceToTarget(PositionWristsmoothed,IndexA0,IndexA25,IndexAM25,Failed_trial,a,StartStop)
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
for i = 1:length(IndexA0)
   
if(StartStop(IndexA0(i),1) == 0 && (~ismember(IndexA0(i),Failed_trial(a,2:end))))
   StartStop(IndexA0(i),1) = 1
   StartStop(IndexA0(i),2)=1
end
    if( ~ismember(IndexA0(i),Failed_trial(a,2:end)) )
        if isnan(StartStop(IndexA0(i),1))
            WristDistanceA0(i) = NaN;
        else
WristDistanceA0(i) = pdist([PositionWristsmoothed(StartStop(IndexA0(i),1),[1 3],IndexA0(i));PositionWristsmoothed(StartStop(IndexA0(i),2),[1 3],IndexA0(i))]);
        end
        
    else
        WristDistanceA0(i) = NaN;
    end
end

for i = 1:length(IndexA25)
     if StartStop(IndexA25(i),1) == 0 && (~ismember(IndexA25(i),Failed_trial(a,2:end)))
   StartStop(IndexA25(i),1) = 1
   StartStop(IndexA25(i),2)=1
end
    if ~ismember(IndexA25(i),Failed_trial(a,2:end)) ||~isnan(StartStop(IndexA25(i),1))
         if isnan(StartStop(IndexA25(i),1))
            WristDistanceA25(i) = NaN;
         else
WristDistanceA25(i) = pdist([PositionWristsmoothed(StartStop(IndexA25(i),1),[1 3],IndexA25(i));PositionWristsmoothed(StartStop(IndexA25(i),2),[1 3],IndexA25(i))]);

         end
         else
        WristDistanceA25(i) = NaN;
    end
end
for i = 1:length(IndexAM25)
     if StartStop(IndexAM25(i),1) == 0 && (~ismember(IndexA0(i),Failed_trial(a,2:end)))
        
   StartStop(IndexAM25(i),1) = 1;
   StartStop(IndexAM25(i),2)=1;
end
    if ~ismember(IndexAM25(i),Failed_trial(a,2:end)) || ~isnan(StartStop(IndexAM25(i),1))
    if isnan(StartStop(IndexAM25(i),1))
            WristDistanceAM25(i) = NaN;
    else
WristDistanceAM25(i) = pdist([PositionWristsmoothed(StartStop(IndexAM25(i),1),[1 3],IndexAM25(i));PositionWristsmoothed(StartStop(IndexAM25(i),2),[1 3],IndexAM25(i))]);
    end
    else
        WristDistanceAM25(i) = NaN;
    end
end
  for i = 1:length(StartStop) % to have directly the whole distance computed without discrimintaing the angle
     if StartStop(i,1) == 0 && (~ismember(i,Failed_trial(a,2:end)))
        
   StartStop(i,1) = 1;
   StartStop(i,2)=1;
end
    if ~ismember(i,Failed_trial(a,2:end)) || ~isnan(StartStop(i,1))
    if isnan(StartStop(i,1))
            WristDistanceTot(i) = NaN;
    else
WristDistanceTot(i) = pdist([PositionWristsmoothed(StartStop(i,1),[1 3],i);PositionWristsmoothed(StartStop(i,2),[1 3],i)]);
    end
    else
        WristDistanceTot(i) = NaN;
    end
end

