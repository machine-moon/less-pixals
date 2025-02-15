
#MAKE SURE TO SOURCE THE ENVIRONMENT VARIABLES:
# $ source ./secrets

all: sync run

run:
	@echo "[EXECUTING] Running ${EXECUTABLE} in ${DEST_DIR} on tpu-$(ACCELERATOR_TYPE)."
	@gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
	--zone=${ZONE} \
	--project=${PROJECT_ID} \
	--worker=all \
	--command="source $(DEST_DIR)/$(RUNTIME_SECRETS) && python3 ${DEST_DIR}/${EXECUTABLE}"


kill:
	@echo "[KILLING] Killing all TPU processes."
	@gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
	--zone=${ZONE} \
	--project=${PROJECT_ID} \
	--worker=all \
	--command="cd ${DEST_DIR} && make __kill"

__kill:
	@echo "[KILLING] Killing >> $HOSTNAME << TPU processes."
	@sudo lsof -t /dev/accel0 | xargs sudo kill -9
	@sudo rm /tmp/libtpu_lockfile
	@sudo chmod -R 777 /tmp/tpu_logs
	@echo "Cleanup complete."


# maybe in future running a git push/pull instead of delete/reupload
sync: __delete __sync


__sync:
	@echo "[SYNCING] Syncing ${SRC_DIR} to ${DEST_DIR} on tpu-$(ACCELERATOR_TYPE)."
	@gcloud compute tpus tpu-vm scp --recurse ${SRC_DIR} ${TPU_NAME}:${DEST_DIR} \
	--worker=all \
	--zone=${ZONE} \
	--project=${PROJECT_ID}

__delete:
	@echo "[DELETING] Deleting ${DEST_DIR} on tpu-$(ACCELERATOR_TYPE)."
	@gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
	--zone=${ZONE} \
	--project=${PROJECT_ID} \
	--worker=all \
	--command='rm -rf ${DEST_DIR}'


__install:
	@echo "[INSTALLING] Installing dependencies on tpu-$(ACCELERATOR_TYPE)."
	@gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
	--zone=${ZONE} \
	--project=${PROJECT_ID} \
	--worker=all \
	--command='pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html &&'

ssh0:
	@echo "[SSH] SSHing into worker 0."
	@gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --project ${PROJECT_ID} --worker 0

ssh1:
	@echo "[SSH] SSHing into worker 1."
	@gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone ${ZONE} --project ${PROJECT_ID} --worker 1

.PHONY: all run kill sync __sync __delete ssh0 ssh1