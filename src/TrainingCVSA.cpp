#include "feedback_cvsa/TrainingCVSA.h"


namespace feedback {

TrainingCVSA::TrainingCVSA(void) : CVSA_layout("trainingCVSA"), p_nh_("~") {

    this->pub_ = this->nh_.advertise<rosneuro_msgs::NeuroEvent>("/events/bus", 1);
    this->sub_ = this->nh_.subscribe("/cvsa/neuroprediction/integrated", 1, &TrainingCVSA::on_received_data, this);
}

TrainingCVSA::~TrainingCVSA(void) {}

bool TrainingCVSA::configure(void) {

    /* Bind dynamic reconfigure callback */
    this->recfg_callback_type_ = boost::bind(&TrainingCVSA::on_request_reconfigure, this, _1, _2);
    this->recfg_srv_.setCallback(this->recfg_callback_type_);

    std::string modality;

    /* PARAMETERS FOR THE LAYOUT */
    // Getting classes
    if(this->p_nh_.getParam("classes", this->classes_) == false) {
        ROS_ERROR("[Training_CVSA] Parameter 'classes' is mandatory");
        return false;
    } 
    this->set_nclasses(this->classes_.size());

    // Getting layout positions
    std::string layout;
    if(this->p_nh_.getParam("circlePositions", layout) == true) {
        if (this->str2matrix(layout).size() != this->nclasses_ || this->str2matrix(layout).at(0).size() != 2){
            ROS_ERROR("[Training_CVSA] The provided layout is not correct. It must be a matrix with %d rows and 2 columns", this->nclasses_);
            return false;
        } 
        this->set_circle_positions(this->str2matrix(layout));
    }else{
        ROS_ERROR("[Training_CVSA] Parameter 'circlePositions' is mandatory");
        return false;
    }
    
    // set up the windows layout
	this->setup();


    /* PARAMETER FOR THE TRIAL EXECUTIONS*/
    // Getting thresholds
    if(this->p_nh_.getParam("thresholds", this->thresholds_) == false) {
        ROS_ERROR("[Training_CVSA] Parameter 'thresholds' is mandatory");
        return false;
    } else if(this->thresholds_.size() != this->nclasses_) {
        ROS_ERROR("[Training_CVSA] Thresholds must be the same of the number of classes %d", this->nclasses_);
        return false;
    }

    // Getting trials per class
    if(this->p_nh_.getParam("trials", this->trials_per_class_) == false) {
        ROS_ERROR("[Training_CVSA] Parameter 'trials' is mandatory");
        return false;
    } else if(this->trials_per_class_.size() != this->nclasses_) { 
        ROS_ERROR("[Training_CVSA] Number of trials per class must be provided for each class");
        return false;
    }
    
    // Getting modality 
    if(this->p_nh_.getParam("modality", modality) == false) {
        ROS_ERROR("[Training_CVSA] Parameter 'modality' is mandatory");
        return false;
    }

    if(modality.compare("calibration") == 0) {
        this->modality_ = Modality::Calibration;
    } else if(modality.compare("evaluation") == 0) {
        this->modality_ = Modality::Evaluation;
    } else {
        ROS_ERROR("Unknown modality provided");
        return false;
    }

    // Getting fake rest class
    this->p_nh_.param("fake_rest", this->fake_rest_, false);
    ROS_WARN("[Training_CVSA] Fake rest is %s", this->fake_rest_ ? "enabled" : "disabled");

    /* PARAMETER FOR THE SOUND FEEDBACK*/
    // Getting parameters for audio feedback
    if(this->p_nh_.getParam("audio_path", this->audio_path_) == false) {
        ROS_ERROR("[Training_CVSA] Parameter 'audio_feedback' is mandatory");
        return false;
    }
    if(this->modality_ == Modality::Evaluation){
        if(this->p_nh_.getParam("init_percentual", this->init_percentual_) == false) {
            ROS_ERROR("[Training_CVSA] Parameter 'init_percentual' is mandatory");
            return false;
        }
        if(this->init_percentual_.size() != this->nclasses_ ) {
            ROS_ERROR("[Training_CVSA] Parameter 'init_percentual' must have the same size of 'classes'");
            return false;
        }else if(static_cast<float>(std::accumulate(this->init_percentual_.begin(), this->init_percentual_.end(), 0.0)) != 1.0f){
            ROS_ERROR("[Training_CVSA] Parameter 'init_percentual' must sum to 1.0, it is %f", std::accumulate(this->init_percentual_.begin(), this->init_percentual_.end(), 0.0));
            return false;
        }
    }
    this->p_nh_.param("audio_cue", this->audio_cue_, false);
    ROS_WARN("[Training_CVSA] Audio cue is %s", this->audio_cue_ ? "enabled" : "disabled");

    /* PARAMETER FOR POSITIVE FEEDBACK*/
    this->p_nh_.param("positive_feedback", this->positive_feedback_, false);
    ROS_WARN("[Training_CVSA] Positive feedback is %s", this->positive_feedback_ ? "enabled" : "disabled");

    /* PARAMETER FOR ROBOT CONTROL */
    this->p_nh_.param("robot_control", this->robot_control_, false);
    ROS_WARN("[Training_CVSA] Robot control is %s", this->robot_control_ ? "enabled" : "disabled");
    if(this->robot_control_){
        this->srv_robot_moving_ = this->nh_.serviceClient<std_srvs::Trigger>("/cvsa/robot_motion");
    }

    /* PARAMETER FOR THE IMU */
    if(this->p_nh_.getParam("imu", this->imu_) == false) {
        ROS_ERROR("[Training_CVSA] Parameter 'imu' is mandatory");
        return false;
    }
    ROS_INFO("[Training_CVSA] IMU is %s", this->imu_ ? "enabled" : "disabled");
    if(this->imu_){
        this->srv_imu_ = this->nh_.serviceClient<std_srvs::Trigger>("/imu_cvsa/receiving_singals");
    }

    /* PARAMETER FOR THE EYE*/
    // eye_ detection
    if(this->p_nh_.getParam("eye_detection", this->eye_detection_) == false) {
        ROS_ERROR("[Training_CVSA] Parameter 'eye_detection' is mandatory");
        return false;
    }
    if(this->eye_detection_){
        this->srv_face_detection_ready_ = this->nh_.serviceClient<std_srvs::Trigger>("cvsa/face_detection_ready");
    }
    // Do or not the eye_calibration
    if(this->p_nh_.getParam("eye_calibration", this->eye_calibration_) == false) {
        ROS_ERROR("[Training_CVSA] Parameter 'eye_calibration' is mandatory");
        return false;
    } 
    if(this->eye_calibration_ == true){
        if(this->p_nh_.getParam("calibration_classes", this->calibration_classes_) == false) {
            ROS_ERROR("[Training_CVSA] Parameter 'calibration_classes' is mandatory since eye_calibration is true");
            return false;
        } 
        if(this->p_nh_.getParam("calibration_positions", layout) == true) {
            if (this->str2matrix(layout).size() != this->calibration_classes_.size() || this->str2matrix(layout).at(0).size() != 2){
                ROS_ERROR("[Training_CVSA] The provided layout for calibration_positions is not correct. It must be a matrix with %ld rows and 2 columns", this->calibration_classes_.size());
                return false;
            } 
            this->calibration_positions_ = this->str2matrix(layout);
        }else{
            ROS_ERROR("[Training_CVSA] Parameter 'calibration_positions' is mandatory since eye_calibration is true");
            return false;
        }
        if(this->p_nh_.getParam("max_trials_per_class", this->max_trials_per_class_) == false) {
            ROS_ERROR("[Training_CVSA] Parameter 'max_trials_per_class' is mandatory since eye_calibration is true");
            if(this->max_trials_per_class_.size() < this->nclasses_){
                ROS_ERROR("[Training_CVSA] The number of max_trials_per_class must be greater than the number of classes");
            }
            return false;
        }else{
            int sum;
            for(int i = 0; i < this->trials_per_class_.size(); i++){
                if(this->max_trials_per_class_.at(i) < this->trials_per_class_.at(i)){
                    ROS_ERROR("[Training_CVSA] The number of max_trials_per_class_ must be greater than the trials_per_class_ for each class of the trials per class");
                    return false;
                }
            }
        }
    }
    // do or not the motion eye online
    this->p_nh_.param("eye_motion_online", this->eye_motion_online_, false);
    if(this->eye_motion_online_){
        this->srv_repeat_trial_ = this->nh_.advertiseService("cvsa/repeat_trial", &TrainingCVSA::on_repeat_trial, this);
        this->pub_trials_keep_ = this->nh_.advertise<feedback_cvsa::Trials_to_keep>("cvsa/trials_keep", 1);
    }


    // Getting duration parameters
    ros::param::param("~duration/begin",            this->duration_.begin,             5000);
    ros::param::param("~duration/start",            this->duration_.start,             1000);
    ros::param::param("~duration/fixation",         this->duration_.fixation,          2000);
    ros::param::param("~duration/cue",              this->duration_.cue,               1000);
    ros::param::param("~duration/feedback_min",     this->duration_.feedback_min,      4000); // duration of cf
    ros::param::param("~duration/feedback_max",     this->duration_.feedback_max,      5500);
    ros::param::param("~duration/boom",             this->duration_.boom,              1000);
    ros::param::param("~duration/timeout",          this->duration_.timeout,          10000); // duration of cf
    ros::param::param("~duration/iti",              this->duration_.iti,                100);
    ros::param::param("~duration/end",              this->duration_.end,               2000);
    ros::param::param("~duration/calibration",      this->duration_.calibration,       2000);


    // Setting parameters
    if(this->modality_ == Modality::Calibration) {
        this->mindur_active_ = this->duration_.feedback_min;
        this->maxdur_active_ = this->duration_.feedback_max;
    } else {
        this->mindur_active_ = this->duration_.timeout;
        this->maxdur_active_ = this->duration_.timeout;
    }

    for(int i = 0; i < this->nclasses_; i++) {
        this->trialsequence_.addclass(this->classes_.at(i), this->trials_per_class_.at(i), this->mindur_active_, this->maxdur_active_);
    }

    if(this->fake_rest_){
        int n_fake_rest = static_cast<int>(std::accumulate(this->trials_per_class_.begin(), this->trials_per_class_.end(), 0) / this->nclasses_);
        this->trialsequence_.addclass(Events::Fake_rest, n_fake_rest, this->mindur_active_, this->maxdur_active_);
    }
    
    ROS_INFO("[Training_CVSA] Total number of classes: %ld", this->classes_.size());
    ROS_INFO("[Training_CVSA] Total number of trials:  %d", this->trialsequence_.size());
    ROS_INFO("[Training_CVSA] Trials have been randomized");

    return true;

}

int TrainingCVSA::class2direction(int eventcue) {

    auto it = find(this->classes_.begin(), this->classes_.end(), eventcue);
    
    if(it != this->classes_.end())
        return int(it - this->classes_.begin());

    return -1;
}

int TrainingCVSA::class2index(int eventcue) {

    auto it = find(this->classes_.begin(), this->classes_.end(), eventcue);
    int idx = -1;

    if(it != this->classes_.end()){
        idx = std::distance(this->classes_.begin(), it);
    }else{
        ROS_ERROR("[Training_CVSA] Class %d not found", eventcue);
    }

    return idx;
}

float TrainingCVSA::direction2threshold(int index) {

	if(index != -1) {
		return this->thresholds_[index];
	} else {
		ROS_ERROR("[Training_CVSA] Unknown direction");
		return -1;
	}
}

std::vector<std::vector<float>> TrainingCVSA::str2matrix(const std::string& str) {
    std::vector<std::vector<float>> matrix;
    std::istringstream iss(str);
    std::string row_str;
    while (std::getline(iss, row_str, ';')) {
        std::istringstream row_ss(row_str);
        float value;
        std::vector<float> row_vector;
        while (row_ss >> value) {
            row_vector.push_back(value);
        }
        matrix.push_back(row_vector);
    }

    return matrix;
}

void TrainingCVSA::on_received_data(const rosneuro_msgs::NeuroOutput& msg) {

    // Check if the incoming message has the provided classes
    bool class_not_found = false;
    std::vector<int> msgclasses = msg.decoder.classes;

    // Check that the incoming classes are the ones provided
    for(auto it = msgclasses.begin(); it != msgclasses.end(); ++it) {
        auto it2 = std::find(this->classes_.begin(), this->classes_.end(), *it);
        if(it2 == this->classes_.end()) {
            class_not_found = true;
            break;
        }
    }

    if(class_not_found == true) {
        ROS_WARN_THROTTLE(5.0f, "[Training_CVSA] The incoming neurooutput message does not have the provided classes");
        return;
    }

    // Set the new incoming data
    this->current_input_ = msg.softpredict.data;

    //std::cout << "Received data: " << this->current_input_[0] << " " << this->current_input_[1] << std::endl;  
}

bool TrainingCVSA::on_repeat_trial(feedback_cvsa::Repeat_trial::Request &req, feedback_cvsa::Repeat_trial::Response &res) {
    this->trial_ok_ = 0;
    int class2repeat = req.class2repeat;
    auto it = std::find(this->classes_.begin(), this->classes_.end(), class2repeat);
    if(it != this->classes_.end()) {
        int idx_class = std::distance(this->classes_.begin(), it);
        if(this->trials_per_class_.at(idx_class) <= this->max_trials_per_class_.at(idx_class)){
            this->trials_per_class_.at(idx_class) = this->trials_per_class_.at(idx_class) + 1;
            this->trialsequence_.addtrial(class2repeat, this->mindur_active_, this->maxdur_active_);
            res.success = true;
            return true;
        }
        
    }
    
    res.success = false;
    return false;
    
}


void TrainingCVSA::run(void) {

    if(this->eye_calibration_ || this->eye_detection_){
        ROS_INFO("[Training_CVSA] Waiting for the camera to be ready");
        this->srv_face_detection_ready_.waitForExistence();
    }
    if(this->robot_control_){
        ROS_INFO("[Training_CVSA] Waiting for the robot motion service to be ready");
        this->srv_robot_moving_.waitForExistence();
    }
    if(this->imu_){
        ROS_INFO("[Training_CVSA] Waiting for the IMU service to be ready");
        this->srv_imu_.waitForExistence();
    }
    std_srvs::Trigger srv_face_detection = std_srvs::Trigger();
    std_srvs::Trigger srv_imu = std_srvs::Trigger();

    while(true){
        if(this->imu_){
            this->srv_imu_.call(srv_imu.request, srv_imu.response);
            if(srv_imu.response.success == false) {
                ROS_WARN_ONCE("[Training_CVSA] IMU is not ready");
                continue;
            }
        }
        if(this->eye_detection_){
            this->srv_face_detection_ready_.call(srv_face_detection.request, srv_face_detection.response);
            if(srv_face_detection.response.success == false) {
                ROS_WARN_ONCE("[Training_CVSA] Camera is not ready");
                continue;
            }
            if(this->eye_calibration_){
                ROS_INFO("[Training_CVSA] Calibration eye started");
                this->eye_calibration();
            }
        }

        ROS_INFO("[Training_CVSA] Protocol BCI started");
        this->show_rings_classes(); // show the rings of each class in the drawing window
        this->bci_protocol();
        break;
    }
    
}

void TrainingCVSA::eye_calibration(void) {

    this->sleep(this->duration_.begin);
    this->show_fixation();
    this->sleep(this->duration_.fixation);
    this->hide_fixation();

    // randomize the order of the calibration classes
    std::vector<int> idx_class;
    for(int i = 0; i < this->calibration_classes_.size(); i++) 
        idx_class.push_back(i);

    std::random_device rnddev;
    std::mt19937 rndgen(rnddev());

    std::shuffle(std::begin(idx_class), std::end(idx_class), rndgen);

    // Start the calibration
    for(int i = 0; i < this->calibration_classes_.size(); i++) {
        this->setevent(Events::StartCalibEye);
        //std::cout << calibration_classes_.at(idx_class.at(i)) << std::endl;
        this->sleep(this->duration_.iti);
        this->setevent(calibration_classes_.at(idx_class.at(i)));
        this->show_calibration(this->calibration_positions_.at(idx_class.at(i)));
        this->sleep(this->duration_.calibration);
        this->hide_calibration();
        this->setevent(calibration_classes_.at(idx_class.at(i)) + Events::Off);
        this->sleep(this->duration_.iti);
        this->setevent(Events::StartCalibEye + Events::Off);
        this->sleep(this->duration_.iti);
    }
    
}

void TrainingCVSA::bci_protocol(void){
    int                    trialnumber;
    int                    trialclass;
    int                    trialduration;
    float                  trialthreshold;
    int                    hitclass;
    int                    boomevent;
    int                    idx_class;
    int                    fake_trialclass;
    int                    trialdirection;
    int                    targethit;
    std::vector<int>       count_results = std::vector<int>(3, 0); // count hit, miss and timeout
    ros::Rate r(this->rate_);
    std::vector<int> idxs_classes(this->nclasses_);
    std::iota(idxs_classes.begin(), idxs_classes.end(), 0);
    std::vector<int> other_idx_classes;
    

    rosneuro::feedback::LinearPilot linearpilot(1000.0f/this->rate_);
    rosneuro::feedback::SinePilot   sinepilot(1000.0f/this->rate_, 0.25f, 0.5f);
    rosneuro::feedback::Autopilot*  autopilot;

    // Begin
    this->sleep(this->duration_.begin);
    
    for(int i = 0; i < this->trialsequence_.size(); i++) {
        // Getting trial information
        trialnumber    = i + 1;
        Trial t = this->trialsequence_.gettrial(i);
        trialclass     = t.classid;
        trialduration  = t.duration;
        
        if(this->fake_rest_ && this->modality_ == Modality::Calibration && trialclass == Events::Fake_rest){
            std::random_device rd; 
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dist(0, this->nclasses_-1);
            idx_class = dist(gen); 
            fake_trialclass = this->classes_.at(idx_class);
            trialdirection = this->class2direction(fake_trialclass);
            trialthreshold = this->direction2threshold(trialdirection);
            other_idx_classes = idxs_classes;
            other_idx_classes.erase(std::remove(other_idx_classes.begin(), other_idx_classes.end(), idx_class), other_idx_classes.end());
        }else{
            idx_class      = this->class2index(trialclass); 
            trialdirection = this->class2direction(trialclass);
            trialthreshold = this->direction2threshold(trialdirection);
        }
        targethit      = -1;
        this->trial_ok_ = 1;

        if(this->modality_ == Modality::Calibration) {
            if(trialclass == Events::Fake_rest && this->fake_rest_){
                autopilot = &sinepilot;
            }else{
                autopilot = &linearpilot;
            }
            autopilot->set(0.0f, trialthreshold, trialduration);
        }

        ROS_INFO("[Training_CVSA] Trial %d/%d (class: %d | duration cf: %d ms)", trialnumber, this->trialsequence_.size(), trialclass, trialduration);
        this->setevent(Events::Start);
        this->sleep(this->duration_.start);
        //this->setevent(Events::Start + Events::Off);

        if(ros::ok() == false || this->user_quit_ == true) break;
        

        /* FIXATION */
        this->setevent(Events::Fixation);
        this->show_fixation();
        this->sleep(this->duration_.fixation);
        this->hide_fixation();
        this->setevent(Events::Fixation + Events::Off);

        if(ros::ok() == false || this->user_quit_ == true) break;


        /* CUE */
        int idx_sampleAudio;
        size_t sampleAudio, bufferAudioSize, n_sampleAudio;
        if(this->fake_rest_ && this->modality_ == Modality::Calibration && trialclass == Events::Fake_rest){
            this->setevent(trialclass);
            this->sleep(this->duration_.iti);
            this->setevent(fake_trialclass);
        }else{
            this->setevent(trialclass);
        }
        this->timer_.tic();
        int c_time;
        if(this->audio_cue_){
            if(this->fake_rest_ && this->modality_ == Modality::Calibration && trialclass == Events::Fake_rest){
                this->loadWAVFile(this->audio_path_ + "/" + std::to_string(fake_trialclass) + ".wav");
            }else{
                this->loadWAVFile(this->audio_path_ + "/" + std::to_string(trialclass) + ".wav");
            }
            this->openAudioDevice();
            this->setAudio(idx_sampleAudio, sampleAudio, bufferAudioSize, n_sampleAudio);
            while(idx_sampleAudio + n_sampleAudio <= this->buffer_audio_full_.size()){       
                this->fillAudioBuffer(idx_sampleAudio, n_sampleAudio, true);
                ao_play(this->device_audio_, reinterpret_cast<char*>(this->buffer_audio_played_.data()), bufferAudioSize * sizeof(short));
            }
            this->closeAudioDevice();
            c_time = this->timer_.toc();
            if(this->duration_.cue - c_time > 0){
                ROS_INFO("[Training_CVSA] Cue added time: %d ms", c_time);
                this->sleep(this->duration_.cue - c_time);
            }
        }else{
            this->show_cue(trialdirection);
            this->sleep(this->duration_.cue);
            this->hide_cue();
        }
        if(this->fake_rest_ && this->modality_ == Modality::Calibration && trialclass == Events::Fake_rest){
            this->sleep(this->duration_.iti);
            this->setevent(fake_trialclass + Events::Off);
        }
        this->setevent(trialclass + Events::Off);
        
        
        if(ros::ok() == false || this->user_quit_ == true) break;


        /* CONTINUOUS FEEDBACK */
        this->timer_.tic();

        // Consuming old messages
        ros::spinOnce();

        // Send cf event
        this->setevent(Events::CFeedback);
        this->show_center();

        // Start the sound feedback
        this->loadWAVFile(this->audio_path_ + "/cf.wav");
        this->openAudioDevice();
        this->setAudio(idx_sampleAudio, sampleAudio, bufferAudioSize, n_sampleAudio);

        // Set up initial probabilities
        this->current_input_ = std::vector<float>(this->nclasses_, 0.0f); 

        while(ros::ok() && this->user_quit_ == false && targethit == -1 && idx_sampleAudio + n_sampleAudio < this->buffer_audio_full_.size()) {

            c_time = this->timer_.toc();
            if(this->modality_ == Modality::Calibration) {
                this->fillAudioBuffer(idx_sampleAudio, n_sampleAudio, false);
                ao_play(this->device_audio_, reinterpret_cast<char*>(this->buffer_audio_played_.data()), bufferAudioSize * sizeof(short));
                if(trialclass == Events::Fake_rest && this->fake_rest_){
                    float step = autopilot->step();
                    if((step <= 0 && this->current_input_[idx_class] == 0.0f && this->current_input_[other_idx_classes[0]] == 0.0f) ||
                       (step >= 0 && this->current_input_[other_idx_classes[0]] > 0.0f) ||
                       (step <= 0 && this->current_input_[idx_class] == 0.0f)){
                        this->current_input_[other_idx_classes[0]] = this->current_input_[other_idx_classes[0]] + (-1)*step; 
                    }else{
                        this->current_input_[idx_class] = this->current_input_[idx_class] + step;
                    }
                    //ROS_INFO("Probabilities: %f %f Thresholds: %f %f", this->current_input_[0], this->current_input_[1], this->thresholds_[0], this->thresholds_[1]);
                }else{
                    this->current_input_[idx_class] = this->current_input_[idx_class] + autopilot->step();
                }
            } else if(this->modality_ == Modality::Evaluation) {
                if(!this->positive_feedback_){
                    this->fillAudioBuffer(idx_sampleAudio, n_sampleAudio, false);
                    ao_play(this->device_audio_, reinterpret_cast<char*>(this->buffer_audio_played_.data()), bufferAudioSize * sizeof(short));
                }else{
                    std::vector<float> input_norm = this->normalize4audio(this->current_input_);
                    auto maxElemIter = std::max_element(input_norm.begin(), input_norm.end());
                    int idx_maxElem = std::distance(input_norm.begin(), maxElemIter);
                    if(idx_maxElem == idx_class){
                        this->fillAudioBuffer(idx_sampleAudio, n_sampleAudio, false);
                        ao_play(this->device_audio_, reinterpret_cast<char*>(this->buffer_audio_played_.data()), bufferAudioSize * sizeof(short));
                    }
                }
                //ROS_INFO("Probabilities: %f %f Thresholds: %f %f", this->current_input_[0], this->current_input_[1], this->thresholds_[0], this->thresholds_[1]);
            }
            
            targethit = this->is_target_hit(this->current_input_,  
                                            c_time, trialduration);

            if(targethit != -1)
                break;
        
            r.sleep();
            ros::spinOnce();
        }
        this->hide_center();
        this->setevent(Events::CFeedback + Events::Off);
        this->closeAudioDevice();
        if(ros::ok() == false || this->user_quit_ == true) break;
        

        /* BOOM */
        if(trialdirection == targethit){
            boomevent = Events::Hit;
        }else if(targethit >= 0 && targethit < this->nclasses_){
            boomevent = Events::Miss;
        }else{
            boomevent = Events::Timeout;
        }
        // for the robot motion
        if(this->robot_control_){
            this->setevent(boomevent);
            this->show_boom(trialdirection, targethit);
            this->timer_.tic();
            std_srvs::Trigger srv;
            while(true){
                this->srv_robot_moving_.call(srv.request, srv.response);
                if(!srv.response.success){
                    break;
                }else{
                    ROS_WARN_ONCE("[Training_CVSA] Robot is moving. Waiting for the robot to stop.");
                }
                this->sleep(500);
            }
            c_time = this->timer_.toc();
            if(c_time < this->duration_.boom){
                this->sleep(this->duration_.boom - c_time);
                ROS_INFO("[Training_CVSA] Boom added time: %d ms", c_time);
            }
            this->hide_boom();
            this->setevent(boomevent + Events::Off);
        }else{
            this->setevent(boomevent);
            this->show_boom(trialdirection, targethit);
            this->sleep(this->duration_.boom);
            this->hide_boom();
            this->setevent(boomevent + Events::Off);
        }

        switch(boomevent) {
            case Events::Hit:
                count_results[0] = count_results[0]+1;
                ROS_INFO("[Training_CVSA] Target hit");
                break;
            case Events::Miss:
                count_results[1] = count_results[1]+1;
                ROS_INFO("[Training_CVSA] Target miss");
                break;
            case Events::Timeout:
                count_results[2] = count_results[2]+1;
                ROS_INFO("[Training_CVSA] Timeout reached. Time elapsed: %d, time duration: %d", c_time, trialduration);
                break;
        }


        /* FINISH the trial */
        this->setevent(Events::Start + Events::Off);

        if(ros::ok() == false || this->user_quit_ == true) break;
        this->trials_keep_.push_back(this->trial_ok_);

        // Inter trial interval
        this->reset();
        this->sleep(this->duration_.iti);

        if(ros::ok() == false || this->user_quit_ == true) break;

    }

    // Print accuracy
    ROS_INFO("[Training_CVSA] Hit: %d, Miss: %d, Timeout: %d", count_results[0], count_results[1], count_results[2]);

    // End
    if(user_quit_ == false)
        this->sleep(this->duration_.end);
    ROS_INFO("[Training_CVSA] Protocol ended");

    // Publish the trials keep
    if(this->eye_motion_online_){
        feedback_cvsa::Trials_to_keep msg;
        msg.trials_to_keep = this->trials_keep_;
        this->pub_trials_keep_.publish(msg);
    }
}

std::vector<float> TrainingCVSA::normalize4audio(std::vector<float>& input) {
    std::vector<float> input_norm(input.size());
    
    // if the value of a class is lower than the initial_percentual then its output is 0
    // otherwise it must be normalized between initial_percentual and threshold
    for(int i = 0; i < input.size(); i++) {
        if(input.at(i) <= this->init_percentual_.at(i)) {
            input_norm.at(i) = 0.0f;
        } else {
            input_norm.at(i) = (input.at(i) - this->init_percentual_.at(i)) / (this->thresholds_.at(i) - this->init_percentual_.at(i));
        }
    }

    return input_norm;
}

void TrainingCVSA::fillAudioBuffer(int& idx_sampleAudio, const size_t& n_sampleAudio, bool cue) {

    std::vector<float> input_norm = std::vector<float>(this->nclasses_, 0.0f);

    if(cue){
        input_norm = std::vector<float>(this->nclasses_, 1.0f);
    }else if(this->modality_ == Modality::Evaluation && !cue){
        input_norm = this->normalize4audio(this->current_input_);
    }else if(this->modality_ == Modality::Calibration && !cue){
        input_norm = this->current_input_;
    }

    for(int i = 0; i < n_sampleAudio * this->channels_audio_; i += this->channels_audio_) {
        for(int j = 0; j < this->channels_audio_; j++) {
            this->buffer_audio_played_.at(i+j) = this->buffer_audio_full_.at(i+j+idx_sampleAudio*this->channels_audio_) * input_norm.at(j);   
        }
    }
    idx_sampleAudio += n_sampleAudio;
}

void TrainingCVSA::loadWAVFile(const std::string& filename) {
    SF_INFO sfInfo;
    SNDFILE *file = sf_open(filename.c_str(), SFM_READ, &sfInfo);
    if (!file) {
        ROS_ERROR("[Training_CVSA] Error opening WAV file: %s", filename.c_str());
        return;
    }

    this->channels_audio_ = sfInfo.channels;
    this->sampleRate_audio_ = sfInfo.samplerate;

    if(this->nclasses_ != this->channels_audio_ && filename.find("cf.wav") != std::string::npos) {
        ROS_WARN("[Training_CVSA] The number of classes (%d) is different of the number of channels of the audio feedback (%d)", this->nclasses_, this->channels_audio_);
    }

    this->buffer_audio_full_.resize(sfInfo.frames * sfInfo.channels);
    sf_read_short(file, this->buffer_audio_full_.data(), this->buffer_audio_full_.size());

    sf_close(file);
}

void TrainingCVSA::setAudio(int& idx_sampleAudio, size_t& sampleAudio, size_t& bufferAudioSize, size_t& n_sampleAudio){
    sampleAudio = this->sampleRate_audio_/this->rate_;
    bufferAudioSize = sampleAudio * this->channels_audio_;
    this->buffer_audio_played_.resize(bufferAudioSize);
    idx_sampleAudio = 0;
    n_sampleAudio = this->sampleRate_audio_/this->rate_;
}

void TrainingCVSA::openAudioDevice(){
    ao_sample_format aoFormat;
    int defaultDriver;

    // Initialize libao
    ao_initialize();
    defaultDriver = ao_default_driver_id();

    // Set format
    aoFormat.bits = 16;
    aoFormat.channels = this->channels_audio_;
    aoFormat.rate = this->sampleRate_audio_;
    aoFormat.byte_format = AO_FMT_NATIVE;
    aoFormat.matrix = nullptr;

    // Open device
    this->device_audio_ = ao_open_live(defaultDriver, &aoFormat, nullptr);
    if (this->device_audio_ == nullptr) {
        ROS_ERROR("[Training_CVSA] Error opening device audio.");
        return;
    }
}

void TrainingCVSA::closeAudioDevice(void) {
    ao_close(this->device_audio_);
    ao_shutdown();
}

void TrainingCVSA::setevent(int event) {

    this->event_msg_.header.stamp = ros::Time::now();
    this->event_msg_.event = event;
    this->pub_.publish(this->event_msg_);
}

void TrainingCVSA::sleep(int msecs) {
    std::this_thread::sleep_for(std::chrono::milliseconds(msecs));
}

int TrainingCVSA::is_target_hit(std::vector<float> input, int elapsed, int duration) {

    int target = -1;

    for(int i = 0; i < this->nclasses_; i++) {
        if(input.at(i) >= this->thresholds_.at(i)) { 
            target = i;
            break;
        } else if(elapsed > duration){
            target = CuePalette.size()-1;
            break; 
        }
    }
    
    return target;
}

void TrainingCVSA::on_request_reconfigure(config_cvsa &config, uint32_t level) {

    switch (this->nclasses_)
    {
    case 2:
        if(std::fabs(config.threshold_0 - this->thresholds_[0]) > 0.00001) {
            ROS_WARN("[Training_CVSA] Threshold class %d changed from %f to %f", this->classes_.at(0), this->thresholds_[0], config.threshold_0);
            this->thresholds_[0] = config.threshold_0;
        }
        if(std::fabs(config.threshold_1 - this->thresholds_[1]) > 0.00001) {
            ROS_WARN("[Training_CVSA] Threshold class %d changed from %f to %f", this->classes_.at(1), this->thresholds_[1], config.threshold_1);
            this->thresholds_[1] = config.threshold_1;
        }
        break;
    case 3:
        if(std::fabs(config.threshold_0 - this->thresholds_[0]) > 0.00001) {
            ROS_WARN("[Training_CVSA] Threshold class %d changed from %f to %f", this->classes_.at(0), this->thresholds_[0], config.threshold_0);
            this->thresholds_[0] = config.threshold_0;
        }
        if(std::fabs(config.threshold_1 - this->thresholds_[1]) > 0.00001) {
            ROS_WARN("[Training_CVSA] Threshold class %d changed from %f to %f", this->classes_.at(1), this->thresholds_[1], config.threshold_1);
            this->thresholds_[1] = config.threshold_1;
        }
        if(std::fabs(config.threshold_2 - this->thresholds_[2]) > 0.00001) {
            ROS_WARN("[Training_CVSA] Threshold class %d changed from %f to %f", this->classes_.at(2), this->thresholds_[2], config.threshold_2);
            this->thresholds_[2] = config.threshold_2;
        }
        break;
    case 4:
        if(std::fabs(config.threshold_0 - this->thresholds_[0]) > 0.00001) {
            ROS_WARN("[Training_CVSA] Threshold class %d changed from %f to %f", this->classes_.at(0), this->thresholds_[0], config.threshold_0);
            this->thresholds_[0] = config.threshold_0;
        }
        if(std::fabs(config.threshold_1 - this->thresholds_[1]) > 0.00001) {
            ROS_WARN("[Training_CVSA] Threshold class %d changed from %f to %f", this->classes_.at(1), this->thresholds_[1], config.threshold_1);
            this->thresholds_[1] = config.threshold_1;
        }
        if(std::fabs(config.threshold_2 - this->thresholds_[2]) > 0.00001) {
            ROS_WARN("[Training_CVSA] Threshold class %d changed from %f to %f", this->classes_.at(2), this->thresholds_[2], config.threshold_2);
            this->thresholds_[2] = config.threshold_2;
        }
        if(std::fabs(config.threshold_3 - this->thresholds_[3]) > 0.00001) {
            ROS_WARN("[Training_CVSA] Threshold class %d changed from %f to %f", this->classes_.at(3), this->thresholds_[3], config.threshold_3);
            this->thresholds_[3] = config.threshold_3;
        }
        break;
    default:
        break;
    }
}

} // namespace feedback