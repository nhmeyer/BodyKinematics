function [v_r_,v_x_,v_r_filt,v_x_filt,PeakVelocity,Max_v_r_Index,PeakVelocity_x,Max_v_x_Index,MovementDuration,PeakLatency,StartStop,fiveperc_v_r,fiveperc_v_x,v_r_cut] = Velocity(PositionWristsmoothed,Times,Failed_trial,V_coordinate,a)
%This function calculate the velocity of a vector of position.
%It calculate both the radial velocity, and the velocity in one dimension
%(x or z to choose). After computing the velocity, it is filtered, then the beginning and end of the velocity peak
%is calculated ( = 5% of the peak velocity). %From there the Peak latency,
%and moevment duration are computed
v_r_cut = zeros(length(PositionWristsmoothed(:,:,1)),size(PositionWristsmoothed,3));

for i = 1:size(PositionWristsmoothed,3)
    if ~ismember(i,Failed_trial(a,2:end))
        
                r(:,i) = sqrt(PositionWristsmoothed(:,1,i).^2 + PositionWristsmoothed(:,3,i).^2)';
                v_r_(:,i) = diff(r(1:length(PositionWristsmoothed)-30,i))./diff(Times(1:length(PositionWristsmoothed)-30,i));
                 v_x_(:,i) =diff(PositionWristsmoothed(1:length(PositionWristsmoothed)-30,V_coordinate,i))./diff(Times(1:length(PositionWristsmoothed)-30,i));
               
              
                % Filter velociy  moving average found experimentaly
               
                v_r_filt(:,i) = smooth(v_r_(:,i), 0.02, 'moving');
               v_x_filt(:,i) = smooth(v_x_(:,i), 0.02, 'moving');
               
                  
                % Remove data that have a velocity below 5% of the peak
                % velocity
                [max_v_r,max_v_r_index] = max(v_r_filt(:,i));
                fiveperc_v_r(:,i) =  max_v_r* 0.05;
                PeakVelocity(i,1) = max_v_r;    
               Max_v_r_Index(i,:) = [max_v_r_index i];
            
               [max_v_x,max_v_x_index] = max(v_x_filt(:,i));
                fiveperc_v_x(:,i) =  max_v_x* 0.05;
                PeakVelocity_x(i,1) = max_v_x;    
               Max_v_x_Index(i,:) = [max_v_x_index i];
           idx_ = find(v_r_filt(:,i) > fiveperc_v_r(:,i));
                dif = diff(idx_);
                nonones = find(dif~=1);
                dif(nonones) = 0;
                zpos = find(~[0 dif' 0]);
                [~, grpidx] = max(diff(zpos));
                if ~isempty(idx_) && (( zpos(grpidx+1)-2)~= 0)
                  
                idx = idx_(zpos(grpidx)):idx_(zpos(grpidx+1)-2);%cut the when the velocity go back below
                
           
                %  the treshold
%                 idx(:,i) = idx_(zpos(grpidx),i):length(v_r_filt(:,i)); % cut only the begin and keep then end
                v_r_cut(idx,i) = v_r_filt(idx,i);
                END = find(v_r_cut(:,i)~= 0);
                t = idx;
                StartStop(i,:) = [t(1) t(end)];
                PeakLatency(i,1) = 1/120 * (Max_v_r_Index(i,1)-t(1));
                MovementDuration(i,1) = Times(t(end)) - Times(t(1));
                v_r_cut(find(v_r_cut(:,i) == 0),i)= NaN;
                else
                    v_r_cut(:,i) = NaN;
                    StartStop(i,1:2) = NaN; 
                    PeakLatency(i,:) = NaN;
                    MovementDuration(i,:) = NaN;
                end
                xlim([0 4])
                ylim([0 2])
idx_ =[];idx = [];

    else
    PeakVelocity_x(i,1)=NaN;
    PeakVelocity(i,1) =NaN;
    StartStop(i,1:2) = NaN;    
    PeakLatency(i,:) = NaN;
    MovementDuration(i,:) = NaN;
    end
end
end

