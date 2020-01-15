

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

random_device rd;
default_random_engine gen(rd());
void ParticleFilter::init(double x, double y, double theta, double std[])
{
  num_particles = 350;
  particles.resize(num_particles);

  //Set Gaussian Distribution with mean and stddev
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  for (int i = 0; i < num_particles; i++)
  {
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{

  //Adding noise to movement
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);
  for (int i = 0; i < num_particles; i++)
  {
    //Motion modell
    double theta = particles[i].theta;
    double d_theta_t = yaw_rate * delta_t;
    if (yaw_rate > 0.001)
    {
      particles[i].x += (velocity / yaw_rate) * (sin(theta + d_theta_t) - sin(theta));
      particles[i].y += (velocity / yaw_rate) * (-cos(theta + d_theta_t) + cos(theta));
      particles[i].theta += d_theta_t;
    }
    else
    {
      particles[i].x += velocity * delta_t * cos(theta);
      particles[i].y += velocity * delta_t * sin(theta);
    }
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  //Re-init for resampling
  weights.resize(num_particles);
  for (size_t i = 0; i < particles.size(); i++)
  {
    //Init particle
    particles[i].associations.clear();
    particles[i].sense_x.clear();
    particles[i].sense_y.clear();
    particles[i].weight = 1.0;

    //Transform observations to map coordinate system
    vector<LandmarkObs> mapped_obs;
    mapped_obs = TransformToMapCoords(particles[i], observations);

    //Search closest landmark for every observation at one particle
    for (size_t j = 0; j < mapped_obs.size(); j++)
    {
      vector<double> distances_lm_sm; // distances between landmark and sensor measurements
      for (size_t k = 0; k < map_landmarks.landmark_list.size(); k++)
      {
        //Calc distance between landmark and current particle
        int lm_id = map_landmarks.landmark_list[k].id_i;
        double lm_x = map_landmarks.landmark_list[k].x_f;
        double lm_y = map_landmarks.landmark_list[k].y_f;
        double distance_lm_pt = dist(particles[i].x, particles[i].y, lm_x, lm_y);
        if (distance_lm_pt <= sensor_range)
        {
          // Calc distance between observation and all landmarks within sensor range
          double distance_lm_sm = dist(mapped_obs[j].x, mapped_obs[j].y, lm_x, lm_y);
          distances_lm_sm.push_back(distance_lm_sm);

          //Association for visualization
          particles[i].associations.push_back(lm_id);
          particles[i].sense_x.push_back(lm_x);
          particles[i].sense_y.push_back(lm_y);
        }
        else
        {
          //If landmark is out of range, then distance is inf.
          distances_lm_sm.push_back(99999999.9);
        }
      }
      int closest_lm_index = distance(distances_lm_sm.begin(), min_element(distances_lm_sm.begin(), distances_lm_sm.end()));
      double lm_closest_x = map_landmarks.landmark_list[closest_lm_index].x_f;
      double lm_closest_y = map_landmarks.landmark_list[closest_lm_index].y_f;
      particles[i].weight *= getWeight(lm_closest_x, lm_closest_y, mapped_obs[j].x, mapped_obs[j].y, std_landmark[0], std_landmark[1]);
    }
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample()
{
  //Resampling Wheel Algorithm
  vector<Particle> newParticles;
  int index = rand() % (num_particles); // 0-100
  double beta = 0.0;
  double mw = *max_element(weights.begin(), weights.end());
  for (int i = 0; i < num_particles; i++)
  {
    beta += ((double)rand() / RAND_MAX) * 2.0 * mw;
    while (beta > weights[index])
    {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    particles[i].id = i;
    newParticles.push_back(particles[index]);
  }
  particles = newParticles;
}

vector<LandmarkObs> ParticleFilter::TransformToMapCoords(Particle particle, vector<LandmarkObs> observation)
{
  //From perspective of particle
  double trans_obs_x, trans_obs_y;
  for (size_t o = 0; o < observation.size(); o++)
  {
    trans_obs_x = particle.x + (cos(particle.theta) * observation[o].x) - (sin(particle.theta) * observation[o].y);
    trans_obs_y = particle.y + (sin(particle.theta) * observation[o].x) + (cos(particle.theta) * observation[o].y);
    observation[o].x = trans_obs_x;
    observation[o].y = trans_obs_y;
  }

  return observation;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}