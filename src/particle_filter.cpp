/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {

  default_random_engine gen;

  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  num_particles = 200;

  for (int i = 0; i < num_particles; ++i) {
    double sample_x, sample_y, sample_theta;

    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);

    particles.push_back(Particle{
        .id = i,
        .x = sample_x,
        .y = sample_y,
        .theta = sample_theta,
        .weight = 1.0,
        .associations = {},
        .sense_x = {},
        .sense_y = {}
    });
  }

  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

  default_random_engine gen;

  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  for (int i = 0; i < num_particles; i++) {

    double x, y, theta;

    if (yaw_rate > 0.0001 || yaw_rate < -0.0001) {
      // Turning
      x = particles[i].x +
          velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      y = particles[i].y +
          velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      theta = particles[i].theta + yaw_rate * delta_t;
    } else {
      // Not turning
      x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
      y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
      theta = particles[i].theta;
    }

    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }

}

vector<LandmarkObs>::iterator ParticleFilter::dataAssociation(
    std::vector<LandmarkObs> predicted,
    LandmarkObs &observation
) {
  // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
  //   implement this method and use it as a helper during the updateWeights phase.

  auto nearest = predicted.begin();
  double min = std::numeric_limits<double>::max();

  for (auto it = predicted.begin(); it != predicted.end(); ++it) {
    LandmarkObs & prediction = *it;

    double distance = dist(observation.x, observation.y, prediction.x, prediction.y);

    if (distance < min) {
      min = distance;
      nearest = it;
    }
  }

  return nearest;

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html


  double gauss_norm = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);

  const double sig_x_2(std_landmark[0] * std_landmark[0]);
  const double sig_y_2(std_landmark[1] * std_landmark[1]);

  for (Particle &particle : particles) {
    vector<LandmarkObs> mapObservations;
    transform(
        observations.begin(),
        observations.end(),
        back_inserter(mapObservations),
        [particle](LandmarkObs &landmarkObs) {
          return LandmarkObs{
              .id = landmarkObs.id,
              .x = particle.x + (landmarkObs.x * cos(particle.theta) - landmarkObs.y * sin(particle.theta)),
              .y = particle.y + (landmarkObs.x * sin(particle.theta) + landmarkObs.y * cos(particle.theta))
          };
        }
    );

    vector<LandmarkObs> landmarksInSensorRange;
    transform(
        map_landmarks.landmark_list.begin(),
        map_landmarks.landmark_list.end(),
        back_inserter(landmarksInSensorRange),
        [](Map::single_landmark_s &landmark) {
          return LandmarkObs{
              .id = landmark.id_i,
              .x = landmark.x_f,
              .y = landmark.y_f
          };
        }
    );

    landmarksInSensorRange.erase(
        remove_if(
            landmarksInSensorRange.begin(),
            landmarksInSensorRange.end(),
            [particle, sensor_range](LandmarkObs &landmarkObs) {
              return dist(particle.x, particle.y, landmarkObs.x, landmarkObs.y) > sensor_range;
            }
        ),
        landmarksInSensorRange.end()
    );

    double weight = 1.0;

    for (LandmarkObs &observation : mapObservations) {
      LandmarkObs & nearest = *dataAssociation(landmarksInSensorRange, observation);
      double dx = observation.x - nearest.x;
      double dy = observation.y - nearest.y;
      double exponent = ((dx * dx) / (2.0 * sig_x_2)) + ((dy * dy) / (2.0 * sig_y_2));

      weight *= gauss_norm * exp(-1.0 * exponent);
    }

    particle.weight = weight;
  }
}

void ParticleFilter::resample() {
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  vector<double> weights(particles.size());

  for (int i = 0; i < particles.size(); i++) {
    weights[i] = particles[i].weight;
  }

  default_random_engine gen;
  std::discrete_distribution<int> dist(weights.begin(), weights.end());

  std::vector<Particle> resampled(particles.size());
  for (int i = 0; i < particles.size(); i++) {
    resampled[i] = particles[dist(gen)];
  }

  particles = resampled;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x,
                                         std::vector<double> sense_y) {
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
