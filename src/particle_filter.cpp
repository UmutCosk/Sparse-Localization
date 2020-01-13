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
    particles[i].weight = 1 / num_particles;
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

  //For any Particle
  for (size_t i = 0; i < particles.size(); i++)
  {
    double theta = particles[i].theta;
    double x = particles[i].x;
    double y = particles[i].y;
    //Transform observations to map coords specific to current particle orientation
    for (size_t o = 0; o < observations.size(); o++)
    {
      observations[o].x = cos(theta) * observations[o].x - sin(theta) * observations[o].x + x;
      observations[o].y = sin(theta) * observations[o].y + cos(theta) * observations[o].y + y;
      std::vector<double> temp_distances;
      //Calculate the distance between current observation and all landmarks
      for (size_t j = 0; j < predicted.size(); j++)
      {
        float distance = sqrt(pow(observations[i].x - predicted[j].x, 2) + pow(observations[i].y - predicted[j].y, 2));
        temp_distances.push_back(distance);
      }
      //Association step: Calculate minimum distance between current observation and all landmarks
      int minElementIndex = std::min_element(temp_distances.begin(), temp_distances.end()) - temp_distances.begin();
      int minElement = *std::min_element(temp_distances.begin(), temp_distances.end());
      //Get id of associated landmark for current particle
      particles[i].associations.push_back(predicted[minElementIndex].id);
    }
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
    for (size_t o = 0; o < observations.size(); o++)
    {
      //Get ID of first landmark which was associated with the observations
      int id = particles[i].associations[o];
      float mean_x = map_landmarks.landmark_list[id].x_f;
      float mean_y = map_landmarks.landmark_list[id].y_f;
      float std_x = std_landmark[0];
      float std_y = std_landmark[1];
    }
  }
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