# Simulate DDNM schedule to understand loop behavior
import sys
sys.path.append('.')
from guided_diffusion.scheduler import get_schedule_jump

# Same parameters as in config
params = {
    't_T': 25,
    'n_sample': 1,
    'jump_length': 5,
    'jump_n_sample': 2
}

times = get_schedule_jump(**params)
time_pairs = list(zip(times[:-1], times[1:]))

print(f"=== DDNM Schedule Analysis ===")
print(f"Total timesteps in schedule: {len(times)}")
print(f"Total time pairs (loop iterations): {len(time_pairs)}")
print(f"\nFirst 10 time pairs:")
for i, (t_last, t_cur) in enumerate(time_pairs[:10]):
    direction = "↓ forward" if t_cur < t_last else "↑ backward jump"
    print(f"  {i}: t={t_last} → t={t_cur} {direction}")

print(f"\nLast 10 time pairs:")
for i, (t_last, t_cur) in enumerate(time_pairs[-10:], start=len(time_pairs)-10):
    direction = "↓ forward" if t_cur < t_last else "↑ backward jump"
    print(f"  {i}: t={t_last} → t={t_cur} {direction}")

print(f"\n=== Key Questions ===")
print(f"1. Last iteration t_last: {time_pairs[-1][0]}")
print(f"2. Last iteration t_cur: {time_pairs[-1][1]}")
print(f"3. Which t values trigger progressive save (t%5==0)?")

save_timesteps = []
for t_last, t_cur in time_pairs:
    if t_last % 5 == 0:
        save_timesteps.append(t_last)

print(f"   Progressive saves at t_last: {sorted(set(save_timesteps))}")
print(f"   Total saves: {len(save_timesteps)} (including duplicates)")

print(f"\n4. How many times is t=0 visited?")
t0_visits = [i for i, (t_last, t_cur) in enumerate(time_pairs) if t_last == 0]
print(f"   t=0 visited {len(t0_visits)} times at iterations: {t0_visits}")

print(f"\n5. Final x0_t in finalresult comes from:")
print(f"   Iteration {len(time_pairs)-1}: t_last={time_pairs[-1][0]}")
