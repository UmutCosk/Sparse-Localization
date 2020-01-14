/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

default_random_engine gen;
void ParticleFilter::init(double x, double y, double theta, double std[])
{
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100; // TODO: Set the number of particles
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

  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  for (int i = 0; i < num_particles; i++)
  {
    //Movement
    double theta = particles[i].theta;
    double d_theta_t = yaw_rate * delta_t;
    particles[i].x += (velocity / yaw_rate) * (sin(theta + d_theta_t) - sin(theta));
    particles[i].y += (velocity / yaw_rate) * (-cos(theta + d_theta_t) + cos(theta));
    particles[i].theta += d_theta_t;

    //Adding Noise
    normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
    normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
    normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations)
{
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (size_t o = 0; o < observations.size(); o++)
  {
    std::vector<double> temp_distances;
    //Calculate the distance between current observation and landmarks
    for (size_t j = 0; j < predicted.size(); j++)
    {
      double distance = dist(observations[o].x, observations[o].y, predicted[j].x, predicted[j].y);
      temp_distances.push_back(distance);
    }
    //Calculate minimum distance between current observation and landmarks
    int index_nearest_landmark = std::min_element(temp_distances.begin(), temp_distances.end()) - temp_distances.begin();

    //Set id of current observation to id of closest landmark
    observations[o].id = predicted[index_nearest_landmark].id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  for (size_t i = 0; i < particles.size(); i++)
  {

    //Get every Landmark within sensor_range of current particle
    vector<LandmarkObs> predicted;
    for (size_t m = 0; m < map_landmarks.landmark_list.size(); m++)
    {
      double x_lm = map_landmarks.landmark_list[m].x_f;
      double y_lm = map_landmarks.landmark_list[m].y_f;
      double distance = dist(x_lm, y_lm, particles[i].x, particles[i].y);
      if (distance <= sensor_range)
      {
        LandmarkObs landmark;
        landmark.id = m;
        landmark.x = x_lm;
        landmark.y = y_lm;
        predicted.push_back(landmark);
      }
    }
    //Copy observation vector to be able to pass to dataAssociation
    vector<LandmarkObs> obs_copy;
    //Transform observation into map coordinates
    obs_copy = TransformToMapCoords(particles[i], observations);
    dataAssociation(predicted, obs_copy);

    //Associate Particle
    for (size_t o = 0; o < obs_copy.size(); o++)
    {
      particles[i].associations.push_back(obs_copy[o].id);
      particles[i].sense_x.push_back(obs_copy[o].x);
      particles[i].sense_y.push_back(obs_copy[o].y);
    }

    for (size_t j = 0; j < obs_copy.size(); j++)
    {
      LandmarkObs landmark = GetLandmarkByID(predicted, obs_copy[j].id);
      particles[i].weight *= getWeight(obs_copy[j].x, obs_copy[j].y, landmark.x, landmark.y, std_landmark[0], std_landmark[1]);
    }
  }
}

LandmarkObs ParticleFilter::GetLandmarkByID(vector<LandmarkObs> landmarks_predicted, int id)
{
  for (size_t i = 0; i < landmarks_predicted.size(); i++)
  {
    if (id == landmarks_predicted[i].id)
    {
      return landmarks_predicted[i];
    }
  }
  std::cout << "No Landmark found!" << std::endl;
  LandmarkObs empty;
  return empty;
}

vector<LandmarkObs> ParticleFilter::TransformToMapCoords(Particle particle, vector<LandmarkObs> observation)
{
  vector<LandmarkObs> mapped;
  if (observation.size() > 0)
  {
    for (size_t o = 0; o < observation.size(); o++)
    {
      LandmarkObs landmark;
      landmark.id = observation[o].id;
      landmark.x = cos(particle.theta) * observation[o].x - sin(particle.theta) * observation[0].x + particle.x;
      landmark.y = sin(particle.theta) * observation[o].y + cos(particle.theta) * observation[0].y + particle.y;
      mapped.push_back(landmark);
    }
  }
  else
  {
    return observation;
  }
  return mapped;
}

void ParticleFilter::resample()
{
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
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