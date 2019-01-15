if [ -z "$(git ls-files -m)" ]; then
  echo "All changes are committed"
else
  echo "Aborting, uncommitted changes"
  exit 1
fi

COMMIT=$(git rev-parse HEAD)
echo "Current commit: $COMMIT"


if [ -z "$1" ]
then
  EXPERIMENT="$COMMIT"
else
  EXPERIMENT="${COMMIT}_$1"
fi

echo "Experiment: $EXPERIMENT"
echo "~~~"
cat configs/dev.json
echo "~~~"

read -p "Start experiment? " -n 1 -r

mkdir -p ~/tmp/$EXPERIMENT
cp configs/dev.json ~/tmp/$EXPERIMENT

train_solver \
  config/dev.json \
  --tensorboard_log_dir=~/tmp/tensorboard/`echo $EXPERIMENT`_`now` \
  --solver_save_dir=~/tmp/s4t/$EXPERIMENT \
  --solver_load_dir=~/tmp/s4t/$EXPERIMENT
