import pystan
import pickle
import datetime
import math
import argparse
import os
from hashlib import md5
import numpy as np
import scipy
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

model_code = """
data {
  // high-level details
  int n_days_total; // 1 + number of days before election that we start collecting poll data
                    // make your life easier by setting this divisible by 7?
                    // e.g. election is on the (n_days_total)'th day. 
  int n_wks_total;  // 1 + total number of weeks before election that we start
                    // e.g. election is on the first day of the (n_wks_total)'th week. 
  int n_states;
  int day_to_week_map[n_days_total]; // maps day to week
  
  // historic data
  vector[n_states] hist_dist; // historic poll showings
  vector[n_states] hist_dist_precision;
  
  // poll info
  int n_polls;
  int polls_n_democratic[n_polls]; // n_clinton[i] = number of clinton voters in poll i
  int polls_n_day[n_polls]; // what day was this poll on?
  int polls_n_wk[n_polls]; // what week was this poll on? (weeks start on Tuesdays)
  int polls_n_voters[n_polls]; // how many people voted in this poll?
  int polls_state[n_polls]; // which state was this poll in? 
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  vector[n_days_total - 1] random_walk_noise_delta; // only $n_days_total - 1$ noise elements are
                                                    // generated b/c election day noise is set to 0)
  matrix[n_wks_total - 1, n_states] random_walk_noise_beta; // (election week noise is set to 0)
  vector[n_states] final_betas_from_standard_normal;
  real<lower=0> sigma_beta;
  real<lower=0> sigma_delta;
}

transformed parameters{
  matrix[n_wks_total, n_states] beta;
  vector[n_days_total] delta;
  vector[n_states] prev_week = rep_vector(0, n_states);
  vector[n_polls] poll_pi_logit;
  matrix[n_days_total, n_states] pi_logit;
  
  real prev_value = 0;
  delta[n_days_total] = 0;
  // let's simulate the random walk for the national trend, delta
  for (i in 1:(n_days_total-1)){
    delta[n_days_total - i] = prev_value + random_walk_noise_delta[n_days_total-i]*sigma_delta;
    prev_value = delta[n_days_total - i];
  }
  
  // set up the final betas:
  for (i in 1:n_states){
    beta[n_wks_total][i] = final_betas_from_standard_normal[i]*hist_dist_precision[i] + hist_dist[i];
    prev_week = to_vector(beta[n_wks_total]);
  }
  
  // let's simulate the random walk for the state-by-state trend, beta.
  // to save computational time, we compute values for beta only weekly. 
  for (i in 1:(n_wks_total - 1)){
    for (j in 1:(n_states)){
      beta[n_wks_total - i][j] = random_walk_noise_beta[i][j]*sigma_beta + prev_week[j];
    }
    prev_week = to_vector(beta[n_wks_total - i]);
  }
  
  // let's convert Betas and deltas into pi's
  for (i in 1:(n_days_total)){
    for (j in 1:n_states){
      pi_logit[i][j] = beta[day_to_week_map[i]][j] + delta[i];
    }
  }
  
  // let's fill the vector which connects the daily state-level stats to each poll
  for (i in 1:n_polls){
    poll_pi_logit[i] = pi_logit[polls_n_day[i]][polls_state[i]];
  }
}

model {
  final_betas_from_standard_normal ~ normal(0, 1);
  random_walk_noise_delta ~ normal(0, 1);
  for (wk in 1:(n_wks_total-1))
    random_walk_noise_beta[wk, 1:n_states] ~ normal(0, 1);
  polls_n_democratic ~ binomial_logit(polls_n_voters, poll_pi_logit);
}

generated quantities{
    matrix[n_days_total, n_states] pi;
    for (i in 1:n_days_total) {
      for (j in 1:n_states) {
        pi[i][j] = inv_logit(pi_logit[i][j]);
      }
    }
}
"""

states = ('alabama illinois montana rhode_island alaska indiana nebraska south_carolina arizona iowa nevada '
          + 'south_dakota arkansas kansas new_hampshire tennessee california kentucky new_jersey texas '
          + 'colorado louisiana new_mexico utah maine new_york vermont connecticut maryland north_carolina '
          + 'virginia delaware massachusetts north_dakota washington florida michigan ohio west_virginia georgia '
          + 'minnesota oklahoma wisconsin hawaii mississippi oregon wyoming idaho missouri pennsylvania').split()

states_num_dict = {state: num for num, state in enumerate(states)}

n_states = 50


def logit(x):
    return math.log(x / (1 - x))


def get_2008_data(hist_dist_std, project_root, weeks_before_election):
    """
    returns an output dictionary -
        int polls_n_democratic[n_polls]; // n_clinton[i] = number of democratic voters in poll i
        int polls_n_day[n_polls]; // what day was this poll on?
        int polls_n_wk[n_polls]; // what week was this poll on? (weeks start on Tuesdays)
        int polls_n_voters[n_polls]; // how many people voted in this poll?
        int polls_state[n_polls]; // which state was this poll in?

        int n_days_total;   // 1 + number of days before election that we start collecting poll data
                            // e.g. election is on the (n_days_total)'th day.
        int n_wks_total;  // 1 + total number of weeks before election that we start
        int day_to_week_map[n_days_total]; // maps day to week
    """
    poll_dir = project_root + '/obama2008/'

    # first lets find the first day of our polling.
    election_date = datetime.datetime(2008, 11, 4)
    date1 = election_date
    for state in states:
        with open(os.path.join(poll_dir, state + '.txt')) as file:
            for line in file:
                w = line.split('\t')
                poll_datestring = w[1].split('-')[1].split()[0].split('/')
                poll_date = datetime.datetime(2008, int(poll_datestring[0]), int(poll_datestring[1]))
                date1 = poll_date if poll_date < date1 else date1

    n_days_total = (election_date - date1).days + 1

    # let's create a mapping from day to week (weeks start on Tuesday because election day is a Tuesday)
    week1_start = date1 - datetime.timedelta(date1.weekday() - 1)  # the first Tuesday of the first week we poll
    day_to_week_map = []
    for i in range(0, n_days_total):
        day_to_week_map.append(((date1 + datetime.timedelta(days=i)) - week1_start).days // 7 + 1)
    n_wks_total = max(day_to_week_map)

    # now let's get the poll data.
    polls_n_voters = []
    polls_n_democratic = []
    polls_n_day = []
    polls_n_wk = []
    polls_state = []
    for state_numb, state in enumerate(states):
        with open(os.path.join(poll_dir, state + '.txt')) as file:
            for line in file:
                w = line.split('\t')
                if len(w) < 6:
                    # print('skipping ' + line)
                    continue
                idx_date = 1
                idx_n_voters = 2
                idx_dem = 4 if len(w) == 6 else 5
                idx_repub = 3 if len(w) == 6 else 4
                poll_datestring = w[idx_date].split('-')[1].split()[0].split('/')
                poll_day = (datetime.datetime(2008, int(poll_datestring[0]), int(poll_datestring[1])) - date1).days + 1
                # some polls were taken after election day, so we can throw them out.
                if poll_day >= n_days_total:
                    # print('skipping ' + line)
                    continue
                # sometimes we only want to look at polls up till a certain date:
                if day_to_week_map[poll_day] > n_wks_total - weeks_before_election:
                    # print('skipping ' + line)
                    continue
                voter_count_str = w[idx_n_voters].split()[0]
                if not voter_count_str.isnumeric():
                    # print('skipping ' + line)
                    continue
                voter_count = int(voter_count_str)
                polls_n_voters.append(voter_count)
                polls_n_democratic.append(
                    math.ceil(voter_count * float(w[idx_dem]) / (float(w[idx_dem]) + float(w[idx_repub]))))
                polls_n_day.append(poll_day)
                polls_n_wk.append(day_to_week_map[poll_day - 1])
                # print('added poll for ' + state + ' - state # {}'.format(state_numb))
                polls_state.append(state_numb + 1)

    hist_dist, hist_dist_precision = create_historical_predictions_for_2008(hist_dist_std, project_root)

    return {'polls_n_voters': polls_n_voters,
            'polls_n_democratic': polls_n_democratic,
            'polls_n_day': polls_n_day,
            'polls_n_wk': polls_n_wk,
            'polls_state': polls_state,
            'n_days_total': int(n_days_total),
            'day_to_week_map': day_to_week_map,
            'n_wks_total': n_wks_total,
            'n_polls': len(polls_n_day),
            'n_states': 50,
            'hist_dist': hist_dist,
            'hist_dist_precision': hist_dist_precision}


def create_historical_predictions_for_2008(hist_dist_std, project_root):
    results_2004_file = project_root + "/2004_results.csv"

    results_2004 = [0] * 50
    with open(results_2004_file) as file:
        for line in file:
            w = line.split(',')
            state = w[0].lower().replace(" ", "_")
            if state not in states_num_dict:
                print("For some reason I can't find " + line + " in the states_num_dict")
                continue
            results_2004[states_num_dict[state]] = float(w[7])

    national_2004_dem_ratio = .487576
    # historical data suggests a +6% "home state advantage" - Kerry was Mass and Bush was Texas
    results_2004[states_num_dict['massachusetts']] -= .03
    results_2004[states_num_dict['texas']] += .03
    time_for_change_2008_national_prediction = .543

    hist_dist = [0] * 50
    for i in range(0, 50):
        hist_dist[i] = results_2004[i] + (
                time_for_change_2008_national_prediction - national_2004_dem_ratio)
        if states[i] == 'illinois':  # obama's home state
            hist_dist[i] += .03
        elif states[i] == 'arizona':  # mccain's home state
            hist_dist[i] -= .03
        hist_dist[i] = logit(hist_dist[i])

    hist_dist_precision = [hist_dist_std] * 50  # .0016 from Abramowitz, Forecasting the 2008Presidential...

    return hist_dist, hist_dist_precision


def get_2016_data(hist_dist_std, project_root, weeks_before_election):
    """
    returns an output dictionary -
        int polls_n_democratic[n_polls]; // n_clinton[i] = number of democratic voters in poll i
        int polls_n_day[n_polls]; // what day was this poll on?
        int polls_n_wk[n_polls]; // what week was this poll on? (weeks start on Tuesdays)
        int polls_n_voters[n_polls]; // how many people voted in this poll?
        int polls_state[n_polls]; // which state was this poll in?

        int n_days_total;   // 1 + number of days before election that we start collecting poll data
                            // e.g. election is on the (n_days_total)'th day.
        int n_wks_total;  // 1 + total number of weeks before election that we start
        int day_to_week_map[n_days_total]; // maps day to week
    """
    poll_dir = project_root + '/trump2016/'

    # first lets find the first day of our polling.
    election_date = datetime.datetime(2016, 11, 8)
    date1 = election_date
    for state in states:
        with open(
                os.path.join(poll_dir, '2016-' + state.replace("_", "-") + '-president-trump-vs-clinton.txt')) as file:
            first_line = True
            for line in file:
                if first_line:
                    first_line = False
                    continue
                w = line.split('\t')
                idx_date = 3
                poll_datestring = [int(t) for t in w[idx_date].strip('"').split('-')]
                poll_date = datetime.datetime(poll_datestring[0], poll_datestring[1], poll_datestring[2])
                date1 = poll_date if poll_date < date1 else date1

    n_days_total = (election_date - date1).days + 1

    # let's create a mapping from day to week (weeks start on Tuesday because election day is a Tuesday)
    week1_start = date1 - datetime.timedelta(date1.weekday() - 1)  # the first Tuesday of the first week we poll
    day_to_week_map = []
    for i in range(0, n_days_total):
        day_to_week_map.append(((date1 + datetime.timedelta(days=i)) - week1_start).days // 7 + 1)
    n_wks_total = max(day_to_week_map)

    # now let's get the poll data.
    polls_n_voters = []
    polls_n_democratic = []
    polls_n_day = []
    polls_n_wk = []
    polls_state = []
    for state_numb, state in enumerate(states):
        with open(
                os.path.join(poll_dir, '2016-' + state.replace("_", "-") + '-president-trump-vs-clinton.txt')) as file:
            first_line = True
            for line in file:
                if first_line:
                    first_line = False
                    continue
                w = line.split('\t')
                if len(w) < 13:
                    continue
                idx_date = 3
                idx_n_voters = 5
                idx_dem = 9
                idx_repub = 8
                poll_datestring = [int(t) for t in w[idx_date].strip('"').split('-')]
                poll_day = (datetime.datetime(poll_datestring[0], poll_datestring[1],
                                              poll_datestring[2]) - date1).days + 1
                # some polls were taken after election day, so we can throw them out.
                if poll_day >= n_days_total:
                    continue
                # sometimes we only want to look at polls up till a certain date:
                if day_to_week_map[poll_day] > n_wks_total - weeks_before_election:
                    continue
                voter_count_str = w[idx_n_voters].strip('"')
                if not voter_count_str.isnumeric():
                    continue
                voter_count = int(voter_count_str)
                polls_n_voters.append(voter_count)
                polls_n_democratic.append(
                    math.ceil(voter_count * float(w[idx_dem]) / (float(w[idx_dem]) + float(w[idx_repub]))))
                polls_n_day.append(poll_day)
                polls_n_wk.append(day_to_week_map[poll_day - 1])
                polls_state.append(state_numb + 1)

    hist_dist, hist_dist_precision = create_historical_predictions_for_2016(hist_dist_std, project_root)

    return {'polls_n_voters': polls_n_voters,
            'polls_n_democratic': polls_n_democratic,
            'polls_n_day': polls_n_day,
            'polls_n_wk': polls_n_wk,
            'polls_state': polls_state,
            'n_days_total': int(n_days_total),
            'day_to_week_map': day_to_week_map,
            'n_wks_total': n_wks_total,
            'n_polls': len(polls_n_day),
            'n_states': 50,
            'hist_dist': hist_dist,
            'hist_dist_precision': hist_dist_precision}


def create_historical_predictions_for_2016(hist_dist_std, project_root):
    results_2012_file = project_root + "/2012_results.csv"

    results_2012 = [0] * 50
    with open(results_2012_file) as file:
        for line in file:
            w = line.split(',')
            state = w[0].strip().lower().replace(" ", "_")
            if state not in states_num_dict:
                print("For some reason I can't find " + line + " in the states_num_dict")
                continue
            results_2012[states_num_dict[state]] = float(w[7])
    national_2012_dem_ratio = .487
    # historical data suggests a +6% "home state advantage" - Obama was Hawaii/Illinois and Romney was Mass/Utah
    results_2012[states_num_dict['massachusetts']] += .03
    results_2012[states_num_dict['utah']] += .03
    results_2012[states_num_dict['hawaii']] -= .03
    results_2012[states_num_dict['illinois']] -= .03
    time_for_change_2016_national_prediction = .543

    hist_dist = [0] * 50
    for i in range(0, 50):
        hist_dist[i] = results_2012[i] + (
                time_for_change_2016_national_prediction - national_2012_dem_ratio)
        hist_dist[i] = logit(hist_dist[i])
        # we on't add any bias for home states, because neither politician represented a state at election time.

    hist_dist_precision = [hist_dist_std] * 50  # .014 from Abramowitz

    return hist_dist, hist_dist_precision


def stan_model_cache(model_code, model_name=None, **kwargs):
    """
    Use just as you would `stan`
    source: https://pystan.readthedocs.io/en/latest/avoiding_recompilation.html
    """
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm, code_hash


def fit_model(model, data, iterations, chains, cache_name):
    print('begin fitting model for ' + cache_name)
    try:
        fit = pickle.load(open(cache_name, 'rb'))
    except:
        fit = model.sampling(data=data, iter=iterations, chains=chains)
        with open(cache_name, 'wb') as f:
            pickle.dump(fit, f)
    print('finished fitting model for ' + cache_name)
    return fit


def run_2016(model, iterations, chains, hist_dist_std, code_hash, project_root, weeks_before_election):
    year = 2016
    data_2016 = get_2016_data(hist_dist_std, project_root, weeks_before_election)
    cache_name = 'fit-model_{}-{}-iterations_{}-chains_{}-'.format(year, code_hash, iterations, chains) + \
                 'hist_dist_std_{}-wks_before{}.pkl'.format(hist_dist_std, weeks_before_election)
    fit = fit_model(model, data_2016, iterations, chains, cache_name)
    return fit


def run_2008(model, iterations, chains, hist_dist_std, code_hash, project_root, weeks_before_election):
    year = 2008
    data_2008 = get_2008_data(hist_dist_std, project_root, weeks_before_election)
    cache_name = 'fit-model_{}-{}-iterations_{}-chains_{}-'.format(year, code_hash, iterations, chains) + \
                 'hist_dist_std_{}-wks_before{}.pkl'.format(hist_dist_std, weeks_before_election)
    fit = fit_model(model, data_2008, iterations, chains, cache_name)
    return fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', help='2008 or 2016?')
    parser.add_argument('--chains', type=int)
    parser.add_argument('--hist_dist_std', type=float)
    parser.add_argument('--iterations', type=int)
    parser.add_argument('--project_root', required=True)
    parser.add_argument('--cores_per_chain', default="1")
    parser.add_argument('--saved_fit', default='')
    parser.add_argument('--saved_model', default='')
    parser.add_argument('--weeks_before_election', default=0, type=int)
    args = parser.parse_args()

    cores_per_chain = args.cores_per_chain
    year = args.year
    chains = args.chains
    hist_dist_std = args.hist_dist_std
    iterations = args.iterations
    project_root = args.project_root
    saved_fit = args.saved_fit
    saved_model = args.saved_model
    weeks_before_election = args.weeks_before_election

    os.environ['STAN_NUM_THREADS'] = cores_per_chain

    data_2008 = get_2008_data(hist_dist_std, project_root, weeks_before_election)

    print('Creating model')
    model, code_hash = stan_model_cache(model_code=model_code)
    if year == '2016' and saved_fit == '':
        fit = run_2016(model, iterations, chains, hist_dist_std, code_hash, project_root, weeks_before_election)
    elif year == '2008' and saved_fit == '':
        fit = run_2008(model, iterations, chains, hist_dist_std, code_hash, project_root, weeks_before_election)

    if saved_fit != '':
        # this means we're gonna have to get a saved model:
        with open(saved_model, "rb") as f:
            model = pickle.load(f)
        with open(saved_fit, "rb") as f:
            fit = pickle.load(f)
        print('loaded model and fit')
    pi = fit.extract(permuted=True)['pi']
    # print('begin fixing posterior')
    # for i in range(0, len(pi)):
    #     for j in range(0, len(pi[i])):
    #         for k in range(0, len(pi[i][j])):
    #             pi[i][j][k] = 1/(1 + math.exp(-1*pi[i][j][k]))
    # print('finish fixing posterior')
    # let's get the Pi's just for Floriddaaaaaa!
    # florida is going to be at pi[chains, days, 35]
    state_id = 35
    pi_florida = np.asarray(pi[:, -180:-1, state_id])  # we only want to get the last 180 days of polling
    '''
    pi_florida looks like:
    array([[0.502, 0.501, 0.5,... 0.503], <-chain1, 180 days 
           [0.492, 0.493, 0.501,... 0.513], <-chain2, 180 days 
           [0.502, 0.500, 0.498,... 0.483], <-chain3, 180 days 
           ])

    '''
    days_confidence_intervals = np.zeros((179, 3))  # the second dimension is m, m-h, m+h
    for i in range(0, 179):
        m, hl, hu = mean_confidence_interval(pi_florida[:, i])
        days_confidence_intervals[i, 0] = m
        days_confidence_intervals[i, 1] = hl
        days_confidence_intervals[i, 2] = hu

    # # use pandas/seaborn.
    # state_df = pd.DataFrame(columns=["Days before election", "Democratic vote"])
    # for chain in range(0, len(pi_florida)):
    #     for day in range(0, 180):
    #         state_df.loc[len(state_df)] = [180-day, pi_florida[chain][day]]
    #
    # election_plot = sns.lineplot(x="Days before election", y="Democratic vote", data=state_df)
    # fig = election_plot.get_figure()
    # fig.savefig('florida_prediction_2016_hist_dist_std_0.014.png')

    plt.plot(range(0, len(days_confidence_intervals)), days_confidence_intervals[:, 0], 'k', color='#3F7F4C')
    plt.fill_between(range(0, len(days_confidence_intervals)), days_confidence_intervals[:, 1],
                     days_confidence_intervals[:, 2], alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99',
                     linewidth=0)
    pic_name = 'state_id_{}-fit-model_{}-iterations_{}-chains_{}-'.format(state_id, year, iterations, chains) + \
               'hist_dist_std_{}-wks_before{}.png'.format(hist_dist_std, weeks_before_election)
    plt.savefig(pic_name)
    print('hello')

    weeks = [16, 14, 12, 10, 8, 6, 4, 2, 0]
    sf = np.zeros((50, len(weeks)))  # state_forecasts

    fit_files = []

    for wk_idx, fit_file in enumerate(fit_files):
        with open(fit_file, "rb") as f:
            fit = pickle.load(f)

        pi = fit.extract(permuted=True)['pi']
        for state in states:
            sf[states_num_dict[state], wk_idx] = pi[:, -1, states_num_dict[state]]

    for idx, state in enumerate(states):
        print(
            '{},{},{},{},{},{},{},{},{},{}'.format(state, sf[idx, 0], sf[idx, 1], sf[idx, 2], sf[idx, 3], sf[idx, 4],
                                                   sf[idx, 5], sf[idx, 6], sf[idx, 7], sf[idx, 8]))

    return


def mean_confidence_interval(data, confidence=0.90):
    # cc-sa https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def main2():
    schools_code = """
    data {
        int<lower=0> J; // number of schools
        vector[J] y; // estimated treatment effects
        vector<lower=0>[J] sigma; // s.e. of effect estimates
    }
    parameters {
        real mu;
        real<lower=0> tau;
        vector[J] eta;
    }
    transformed parameters {
        vector[J] theta;
        theta = mu + tau * eta;
    }
    model {
        eta ~ normal(0, 1);
        y ~ normal(theta, sigma);
    }
    """

    schools_dat = {'J': 8,
                   'y': [28, 8, -3, 7, -1, 1, 18, 12],
                   'sigma': [15, 10, 16, 11, 9, 11, 10, 18]}
    sm, _ = stan_model_cache(model_code=schools_code, model_name='testmodel')
    fit = sm.sampling(data=schools_dat, iter=2500, chains=3)

    la = fit.extract(permuted=True)  # return a dictionary of arrays
    mu = la['mu']

    ## return an array of three dimensions: iterations, chains, parameters
    a = fit.extract(permuted=False)

    print(fit)
    fit.plot()
    print(fit)


if __name__ == "__main__":
    main()
