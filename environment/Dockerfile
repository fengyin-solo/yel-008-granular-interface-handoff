FROM python:3.13.15-slim-bookworm@sha256:00faa2debb87529f9f0764e9491d8ba400a3678976616c3bd7cb193745ac20d1
COPY wheels/ /opt/wheels/
COPY generate_data.py /opt/sim/generate_data.py
COPY app/ /app/
RUN mkdir -p /data /output /logs/verifier \
    && python3 /opt/sim/generate_data.py /data/simulation.json /output/state.chk \
    && chmod 0755 /app/simulate.py /logs/verifier
WORKDIR /app
CMD ["sleep", "infinity"]
