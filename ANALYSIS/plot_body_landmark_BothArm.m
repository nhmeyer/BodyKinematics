function plot_body_landmark_BothArm(FileName1,FileName2, PlotIn3D)
%function plot_body_landmark(FileName1, PlotIn3D)
%
% Plots the data of the Body landmark task
% measures the distance between the points
% and returns the CSV data.
% Parameter:
%       FileName1    :   FileName1 to data
%       PlotIn3D    :   false for 2D (x,z) [default] (0)
%                       true for 3D (x,z,y) (1) e.g. plot_body_landmark ('ELO4IN_HEALTHY_20150708_133342.mat',1)
%
% e.g. plot_body_landmark('data.mat')


%index = cross=1
%ring = circle=2
%int = plus=3
%ext = square=4
%elbow = triangle=5

%to correct trials with error e.g.  result.Perceived(8).position = [NaN,NaN,NaN];

 

% (c) by Robert Leeb, 2015-07-10
% -------------
% check input parameters
if ~exist('FileName1'), error('No data file specified, please use:  plot_body_landmark(''data.mat'')');end
if ~exist('PlotIn3D'), PlotIn3D=false;end

%% -------------
% settings
ConnectMarker = {[1 2],[3 4], [1 3],[2 4], [3 5],[4 5]};
LM_list = ['x','o','+','s','v'];

%PlotIn3D = true;
scale = 1000; Unit='[mm]';
disp_shift_str = 22;
FlipXAxis_Val = 1000;
xyz_str ='XYZ';
%% -------------
% load data
disp(' ')
disp(' ')
disp('-----')
disp(['Load ' FileName1])
disp('-----')
load (FileName1)


% display values
if false
    disp('original values')
    if isfield(result,'CalibrationObject')
        disp(['Cal.Obj.  '  ' / ' num2str(result.CalibrationObject(1:3)*scale, '%.1f ')])
    end
    disp(' REAL (in [mm])')
    for k=1:length(result.Real)
       disp(['ID ' num2str(result.Real(k).LandmarkID,'%02d') ' / '  repmat(' ',1, (result.Real(k).LandmarkID-1)*disp_shift_str) num2str(result.Real(k).position*scale, '%.1f ')])
    end
    disp(' PERCEIVED (in [mm])')
    for k=1:length(result.Perceived)
       disp(['ID ' num2str(result.Perceived(k).LandmarkID,'%02d') ' / ' repmat(' ',1, (result.Perceived(k).LandmarkID-1)*disp_shift_str) num2str(result.Perceived(k).position*scale, '%.1f ')])
    end
    disp(' ')
    disp('flipped values')
end
%% plot values
figure
if PlotIn3D
   % hp=plot3(0,0,0);
    xlabel(['x ' Unit]); ylabel(['z ' Unit]); zlabel(['y ' Unit]);
else
  hp=plot(0,0);
%     set(gca,'xtick',[])
%     set(gca,'ytick',[])
  %  xlabel(['x ' Unit]); ylabel(['z ' Unit]);
end
%set(hp,'visible','off')
grid on
hold on
% calibration
if isfield(result,'CalibrationObject')
    pos = result.CalibrationObject(1:3);
    pos(1) = FlipXAxis_Val/scale-pos(1);
    %disp(['Cal.Obj.  '  ' / ' num2str(pos*scale, '%.1f ')])
end
% real
disp(' REAL (in [mm])')
Real_Mat_pos=[];
for k=1:length(result.Real)
    id = result.Real(k).LandmarkID;
    pos = result.Real(k).position;
    pos(1) = FlipXAxis_Val/scale-pos(1);
    %disp(['ID ' num2str(id,'%02d') ' / '  repmat(' ',1, (id-1)*disp_shift_str) num2str(pos*scale, '%.1f ')])
    if PlotIn3D
       % hp=plot3(pos(1)*scale,pos(3)*scale,pos(2)*scale);
    else
        hp=plot(pos(1)*scale,pos(3)*scale);
        hold on
        %viscircles([pos(1)*scale,pos(3)*scale],10,'Color','g');
        hold on
%           set(gca,'xtick',[])
%     set(gca,'ytick',[])
    end
    set(hp,'MarkerSize',10,'LineWidth',2,'Marker',LM_list(id),'Color',[0 1 0])
  
    Real_Mat_pos(id,:)=pos;
end
%perceived
disp(' PERCEIVED (in [mm])')
Per_Mat_pos=[];
Per_cnt_pos=zeros(length(result.Real),1);
for k=1:length(result.Perceived)
    id = result.Perceived(k).LandmarkID;
    pos = result.Perceived(k).position;
    pos(1) = FlipXAxis_Val/scale-pos(1);
    %disp(['ID ' num2str(id,'%02d') ' / ' repmat(' ',1, (id-1)*disp_shift_str) num2str(pos*scale, '%.1f ')])
    if PlotIn3D
       % hp=plot3(pos(1)*scale,pos(3)*scale,pos(2)*scale);
    else
       hpp=plot(pos(1)*scale,pos(3)*scale);
        
%             set(gca,'xtick',[])
%     set(gca,'ytick',[])
    end
   set(hpp,'MarkerSize',10,'LineWidth',2,'Marker',LM_list(id),'Color',[0.5 0.9 1])    
    if ~isnan(pos)
    Per_cnt_pos(id)=Per_cnt_pos(id)+1;
    Per_Mat_pos(id,Per_cnt_pos(id),:)=pos;
    end
end
% average perceived, only over not NaN ones
Per_Mat_pos_mean=[];
Per_Mat_pos_std =[];
for k=1:size(Per_Mat_pos,1)
    Per_Mat_pos_mean(k,:)=squeeze(mean(Per_Mat_pos(k,1:Per_cnt_pos(k),:),2));
    Per_Mat_pos_std(k,:)=squeeze(std(Per_Mat_pos(k,1:Per_cnt_pos(k),:),[],2));
end
for k=1:length(result.Real)
    id = result.Real(k).LandmarkID;
    pos = Per_Mat_pos_mean(id,:);
    %disp(['Avg ' num2str(id,'%02d') ' / '  repmat(' ',1, (id-1)*disp_shift_str) num2str(pos*scale, '%.1f ')])
    
  
       hpa=plot(pos(1)*scale,pos(3)*scale);
          hold on
       % viscircles([pos(1)*scale,pos(3)*scale],50,'Color','r');
        hold on
%          set(gca,'xtick',[])
%     set(gca,'ytick',[]) 
            
   
    set(hpa,'MarkerSize',10,'LineWidth',2,'Marker',LM_list(id),'Color',[1 0 0])
   

end
%MyMarkers = ['x','o','+','s','v']
title(sprintf('Baseline %s',FileName1(1:3)),'interpreter','none')
plot([350;400],[-85;-85],'-k',[350;350],[-85;-35],'-k','LineWidth',2)
text(365,-92, '5 cm', 'HorizontalAlignment','center')
text(325,-58, '5 cm', 'HorizontalAlignment','center')
green = plot(nan,nan,'g');
blue = plot(nan,nan,'b');
red = plot(nan,nan,'r');
Index = plot(nan,nan,'xk');
Ring = plot(nan,nan,'ok');
InternalWrist = plot(nan,nan,'+k');
ExternalWrist = plot(nan,nan,'sk');
Elbow = plot(nan,nan,'vk');
legend([green blue red Index Ring InternalWrist ExternalWrist Elbow],{'real','perceived','average perceived','Index','Ring','Internal Wrist','External Wrist','Elbow'},'AutoUpdate','off','Location','northwest')
%legend([green blue red Index Ring InternalWrist ExternalWrist Elbow])
%legend([Index Ring InternalWrist ExternalWrist Elbow],{'Index','Ring','Internal Wrist','External Wrist','Elbow'})


if PlotIn3D  
    view([-20,-60,30])
end
%% -------------
% plot connecting lines
for k=1:length(ConnectMarker)
    ids=ConnectMarker{k};
    if PlotIn3D
       % hlr=line(Real_Mat_pos(ids,1)*scale,Real_Mat_pos(ids,3)*scale,Real_Mat_pos(ids,2)*scale);
       % hlp=line(Per_Mat_pos_mean(ids,1)*scale,Per_Mat_pos_mean(ids,3)*scale,Per_Mat_pos_mean(ids,2)*scale);
    else
       hlr=line(Real_Mat_pos(ids,1)*scale,Real_Mat_pos(ids,3)*scale);
       hlp=line(Per_Mat_pos_mean(ids,1)*scale,Per_Mat_pos_mean(ids,3)*scale);
         set(gca,'xtick',[])
    set(gca,'ytick',[])
   axis equal
  % xlim([600 800]);
   ylim([400 1500]);
    %set(gca,'DataAspectRatio',[1 1 1])
    end
   set(hlr,'color','g','lineStyle','-','LineWidth',0.5)
   set(hlp,'color','r','lineStyle','-','LineWidth',0.5)
end
disp('-----'); disp(' ')
hold on
disp(' ')
disp(' ')
disp('-----')
disp(['Load ' FileName2])
disp('-----')
load (FileName2)


% display values
if false
    disp('original values')
    if isfield(result,'CalibrationObject')
        disp(['Cal.Obj.  '  ' / ' num2str(result.CalibrationObject(1:3)*scale, '%.1f ')])
    end
    disp(' REAL (in [mm])')
    for k=1:length(result.Real)
       disp(['ID ' num2str(result.Real(k).LandmarkID,'%02d') ' / '  repmat(' ',1, (result.Real(k).LandmarkID-1)*disp_shift_str) num2str(result.Real(k).position*scale, '%.1f ')])
    end
    disp(' PERCEIVED (in [mm])')
    for k=1:length(result.Perceived)
       disp(['ID ' num2str(result.Perceived(k).LandmarkID,'%02d') ' / ' repmat(' ',1, (result.Perceived(k).LandmarkID-1)*disp_shift_str) num2str(result.Perceived(k).position*scale, '%.1f ')])
    end
    disp(' ')
    disp('flipped values')
end
%% plot values

if PlotIn3D
   % hp=plot3(0,0,0);
    xlabel(['x ' Unit]); ylabel(['z ' Unit]); zlabel(['y ' Unit]);
else
  hp=plot(0,0);
%     set(gca,'xtick',[])
%     set(gca,'ytick',[])
  %  xlabel(['x ' Unit]); ylabel(['z ' Unit]);
end
%set(hp,'visible','off')
grid on
hold on
% calibration
if isfield(result,'CalibrationObject')
    pos = result.CalibrationObject(1:3);
    pos(1) = FlipXAxis_Val/scale-pos(1);
    %disp(['Cal.Obj.  '  ' / ' num2str(pos*scale, '%.1f ')])
end
% real
disp(' REAL (in [mm])')
Real_Mat_pos=[];
for k=1:length(result.Real)
    id = result.Real(k).LandmarkID;
    pos = result.Real(k).position;
    pos(1) = FlipXAxis_Val/scale-pos(1);
    %disp(['ID ' num2str(id,'%02d') ' / '  repmat(' ',1, (id-1)*disp_shift_str) num2str(pos*scale, '%.1f ')])
    if PlotIn3D
       % hp=plot3(pos(1)*scale,pos(3)*scale,pos(2)*scale);
    else
        hp=plot(pos(1)*scale,pos(3)*scale);
          hold on
      %  viscircles([pos(1)*scale,pos(3)*scale],10,'Color','g');
        hold on
%           set(gca,'xtick',[])
%     set(gca,'ytick',[])
    end
    set(hp,'MarkerSize',10,'LineWidth',2,'Marker',LM_list(id),'Color',[0 1 0])
  
    Real_Mat_pos(id,:)=pos;
end
%perceived
disp(' PERCEIVED (in [mm])')
Per_Mat_pos=[];
Per_cnt_pos=zeros(length(result.Real),1);
for k=1:length(result.Perceived)
    id = result.Perceived(k).LandmarkID;
    pos = result.Perceived(k).position;
    pos(1) = FlipXAxis_Val/scale-pos(1);
    %disp(['ID ' num2str(id,'%02d') ' / ' repmat(' ',1, (id-1)*disp_shift_str) num2str(pos*scale, '%.1f ')])
    if PlotIn3D
       % hp=plot3(pos(1)*scale,pos(3)*scale,pos(2)*scale);
    else
       hpp=plot(pos(1)*scale,pos(3)*scale);
       
%             set(gca,'xtick',[])
%     set(gca,'ytick',[])
    end
   set(hpp,'MarkerSize',10,'LineWidth',2,'Marker',LM_list(id),'Color',[0.5 0.9 1])    
    if ~isnan(pos)
    Per_cnt_pos(id)=Per_cnt_pos(id)+1;
    Per_Mat_pos(id,Per_cnt_pos(id),:)=pos;
    end
end
% average perceived, only over not NaN ones
Per_Mat_pos_mean=[];
Per_Mat_pos_std =[];
for k=1:size(Per_Mat_pos,1)
    Per_Mat_pos_mean(k,:)=squeeze(mean(Per_Mat_pos(k,1:Per_cnt_pos(k),:),2));
    Per_Mat_pos_std(k,:)=squeeze(std(Per_Mat_pos(k,1:Per_cnt_pos(k),:),[],2));
end
for k=1:length(result.Real)
    id = result.Real(k).LandmarkID;
    pos = Per_Mat_pos_mean(id,:);
    %disp(['Avg ' num2str(id,'%02d') ' / '  repmat(' ',1, (id-1)*disp_shift_str) num2str(pos*scale, '%.1f ')])
    
  
       hpa=plot(pos(1)*scale,pos(3)*scale);
          hold on
       % viscircles([pos(1)*scale,pos(3)*scale],10,'Color','r');
        hold on
%          set(gca,'xtick',[])
%     set(gca,'ytick',[]) 
            
   
    set(hpa,'MarkerSize',10,'LineWidth',2,'Marker',LM_list(id),'Color',[1 0 0])
   

end
%MyMarkers = ['x','o','+','s','v']
% title([FileName1],'interpreter','none')
% plot([640;690],[-85;-85],'-k',[640;640],[-85;-35],'-k','LineWidth',2)
% text(665,-92, '5 cm', 'HorizontalAlignment','center')
% text(625,-58, '5 cm', 'HorizontalAlignment','center')
% green = plot(nan,nan,'g');
% blue = plot(nan,nan,'b');
% red = plot(nan,nan,'r');
% Index = plot(nan,nan,'xk');
% Ring = plot(nan,nan,'ok');
% InternalWrist = plot(nan,nan,'+k');
% ExternalWrist = plot(nan,nan,'sk');
% Elbow = plot(nan,nan,'vk');
% legend([green blue red Index Ring InternalWrist ExternalWrist Elbow],{'real','perceived','average perceived','Index','Ring','Internal Wrist','External Wrist','Elbow'},'AutoUpdate','off','Location','southeast')
%legend([green blue red Index Ring InternalWrist ExternalWrist Elbow])
%legend([Index Ring InternalWrist ExternalWrist Elbow],{'Index','Ring','Internal Wrist','External Wrist','Elbow'})


if PlotIn3D  
    view([-20,-60,30])
end
%% -------------
% plot connecting lines
for k=1:length(ConnectMarker)
    ids=ConnectMarker{k};
    if PlotIn3D
       % hlr=line(Real_Mat_pos(ids,1)*scale,Real_Mat_pos(ids,3)*scale,Real_Mat_pos(ids,2)*scale);
       % hlp=line(Per_Mat_pos_mean(ids,1)*scale,Per_Mat_pos_mean(ids,3)*scale,Per_Mat_pos_mean(ids,2)*scale);
    else
       hlr=line(Real_Mat_pos(ids,1)*scale,Real_Mat_pos(ids,3)*scale);
       hlp=line(Per_Mat_pos_mean(ids,1)*scale,Per_Mat_pos_mean(ids,3)*scale);
         set(gca,'xtick',[])
    set(gca,'ytick',[])
    axis equal
  % xlim([150 900]);ylim([-100 550]);
    %set(gca,'DataAspectRatio',[1 1 1])
    end
   set(hlr,'color','g','lineStyle','-','LineWidth',0.5)
   set(hlp,'color','r','lineStyle','-','LineWidth',0.5)
end
disp('-----'); disp(' ')


