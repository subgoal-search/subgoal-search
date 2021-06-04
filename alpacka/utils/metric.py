import numpy as np


def compute_episode_metrics(episodes):
    returns_ = np.array([episode.return_ for episode in episodes])
    lengths = np.array([episode.transition_batch.reward.shape[0]
                        for episode in episodes])

    solved_rate = sum(
        int(episode.solved) for episode in episodes
        if episode.solved is not None
    ) / len(episodes)

    return dict(
        return_mean=np.mean(returns_),
        return_median=np.median(returns_),
        return_std=np.std(returns_, ddof=1),
        length_mean=np.mean(lengths),
        length_median=np.median(lengths),
        length_std=np.std(lengths, ddof=1),
        solved_rate=solved_rate,
    )


def compute_scalar_statistics(x, prefix=None, with_min_and_max=False):
    prefix = prefix + '_' if prefix else ''
    stats = {}

    stats[prefix + 'mean'] = np.nanmean(x)
    stats[prefix + 'std'] = np.nanstd(x)
    if with_min_and_max:
        stats[prefix + 'min'] = np.nanmin(x)
        stats[prefix + 'max'] = np.nanmax(x)

    return stats
