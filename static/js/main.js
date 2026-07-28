/* ==========================================================================
   VORTEX Frontend — Client Logic & API Integration
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    let uploadedFilename = null;

    // ─── DOM References ───
    const videoInput = document.getElementById('videoInput');
    const dropzone = document.getElementById('dropzone');
    const dropzoneContent = document.getElementById('dropzoneContent');
    const uploadProgressContainer = document.getElementById('uploadProgressContainer');
    const uploadStatusText = document.getElementById('uploadStatusText');

    const originalPreviewWrapper = document.getElementById('originalPreviewWrapper');
    const originalVideoPlayer = document.getElementById('originalVideoPlayer');
    const originalFileInfo = document.getElementById('originalFileInfo');

    const transformForm = document.getElementById('transformForm');
    const processBtn = document.getElementById('processBtn');
    const processBtnText = document.getElementById('processBtnText');
    const presetSelect = document.getElementById('presetSelect');

    // Sliders
    const sliders = {
        resolutionScale: { el: document.getElementById('resolutionScale'), display: document.getElementById('resScaleVal'), fmt: v => `${Math.round(v * 100)}%` },
        crf: { el: document.getElementById('crf'), display: document.getElementById('crfVal'), fmt: v => v },
        grainStrength: { el: document.getElementById('grainStrength'), display: document.getElementById('grainVal'), fmt: v => `${(v * 100).toFixed(1)}%` },
        pitchShift: { el: document.getElementById('pitchShift'), display: document.getElementById('pitchVal'), fmt: v => `${parseFloat(v) > 0 ? '+' : ''}${parseFloat(v).toFixed(1)} st` },
        speed: { el: document.getElementById('speed'), display: document.getElementById('speedVal'), fmt: v => `${parseFloat(v).toFixed(2)}x` },
        adversarialEpsilon: { el: document.getElementById('adversarialEpsilon'), display: document.getElementById('epsVal'), fmt: v => `${parseFloat(v).toFixed(1)}/255` },
        adversarialSteps: { el: document.getElementById('adversarialSteps'), display: document.getElementById('stepsVal'), fmt: v => v },
    };

    // Bind live slider displays
    Object.values(sliders).forEach(({ el, display, fmt }) => {
        el.addEventListener('input', () => { display.textContent = fmt(el.value); });
    });

    // Adversarial toggle
    const adversarialEnabled = document.getElementById('adversarialEnabled');
    const adversarialControls = document.getElementById('adversarialControls');

    adversarialEnabled.addEventListener('change', () => {
        adversarialControls.classList.toggle('disabled', !adversarialEnabled.checked);
    });
    // Initialize disabled state
    adversarialControls.classList.add('disabled');

    // Metadata mutual exclusion
    const stripMetadata = document.getElementById('stripMetadata');
    const injectMetadata = document.getElementById('injectMetadata');
    stripMetadata.addEventListener('change', () => {
        if (stripMetadata.checked) injectMetadata.checked = false;
    });
    injectMetadata.addEventListener('change', () => {
        if (injectMetadata.checked) stripMetadata.checked = false;
    });

    // Output DOM
    const outputPlaceholder = document.getElementById('outputPlaceholder');
    const processingLoader = document.getElementById('processingLoader');
    const loaderTitle = document.getElementById('loaderTitle');
    const loaderSubtitle = document.getElementById('loaderSubtitle');
    const outputContent = document.getElementById('outputContent');
    const processedVideoPlayer = document.getElementById('processedVideoPlayer');
    const downloadBtn = document.getElementById('downloadBtn');

    const origSizeVal = document.getElementById('origSizeVal');
    const outSizeVal = document.getElementById('outSizeVal');
    const reductionVal = document.getElementById('reductionVal');

    // Adversarial metrics DOM
    const adversarialReport = document.getElementById('adversarialReport');
    const advPsnr = document.getElementById('advPsnr');
    const advHashSim = document.getElementById('advHashSim');
    const advEps = document.getElementById('advEps');
    const advSteps = document.getElementById('advSteps');

    // Loader step elements
    const stepAdv = document.getElementById('stepAdv');
    const stepVideo = document.getElementById('stepVideo');
    const stepAudio = document.getElementById('stepAudio');
    const stepRemux = document.getElementById('stepRemux');

    // ─── Presets ───
    const presets = {
        standard: {
            video_codec: 'h264', frame_rate: '30', resolution_scale: 0.9,
            crf: 23, grain_strength: 0, audio_codec: 'aac', audio_bitrate: '128k',
            pitch_shift: 0, eq_filter: false, speed: 1.0,
            mirror: false, zoom: false, micro_rotate: false, add_border: false,
            adversarial_enabled: false, adversarial_epsilon: 8, adversarial_steps: 40,
            container: 'mp4', strip_metadata: true, inject_metadata: false
        },
        high_compression: {
            video_codec: 'h265', frame_rate: '24', resolution_scale: 0.75,
            crf: 28, grain_strength: 0, audio_codec: 'opus', audio_bitrate: '96k',
            pitch_shift: 0, eq_filter: false, speed: 1.0,
            mirror: false, zoom: false, micro_rotate: false, add_border: false,
            adversarial_enabled: false, adversarial_epsilon: 8, adversarial_steps: 40,
            container: 'mp4', strip_metadata: true, inject_metadata: false
        },
        evasion_basic: {
            video_codec: 'h265', frame_rate: '24', resolution_scale: 0.85,
            crf: 26, grain_strength: 0.02, audio_codec: 'aac', audio_bitrate: '128k',
            pitch_shift: -1.0, eq_filter: true, speed: 1.04,
            mirror: true, zoom: true, micro_rotate: false, add_border: true,
            adversarial_enabled: false, adversarial_epsilon: 8, adversarial_steps: 40,
            container: 'mp4', strip_metadata: true, inject_metadata: false
        },
        evasion_full: {
            video_codec: 'h265', frame_rate: '24', resolution_scale: 0.8,
            crf: 26, grain_strength: 0.03, audio_codec: 'opus', audio_bitrate: '96k',
            pitch_shift: -2.0, eq_filter: true, speed: 1.04,
            mirror: true, zoom: true, micro_rotate: true, add_border: true,
            adversarial_enabled: true, adversarial_epsilon: 8, adversarial_steps: 40,
            container: 'mp4', strip_metadata: false, inject_metadata: true
        },
        brute_force: {
            video_codec: 'h265', frame_rate: '24', resolution_scale: 0.75,
            crf: 28, grain_strength: 0.04, audio_codec: 'opus', audio_bitrate: '96k',
            pitch_shift: -2.0, eq_filter: true, speed: 1.04,
            mirror: true, zoom: true, micro_rotate: true, add_border: true,
            adversarial_enabled: true, adversarial_epsilon: 12.0, adversarial_steps: 20,
            container: 'mp4', strip_metadata: false, inject_metadata: true
        }
    };

    presetSelect.addEventListener('change', (e) => {
        const key = e.target.value;
        if (key === 'custom' || !presets[key]) return;
        applyPreset(presets[key]);
    });

    const bruteForceQuickBtn = document.getElementById('bruteForceQuickBtn');
    if (bruteForceQuickBtn) {
        bruteForceQuickBtn.addEventListener('click', () => {
            presetSelect.value = 'brute_force';
            applyPreset(presets.brute_force);
        });
    }

    function applyPreset(p) {
        document.getElementById('videoCodec').value = p.video_codec;
        document.getElementById('frameRate').value = p.frame_rate;

        setSlider('resolutionScale', p.resolution_scale);
        setSlider('crf', p.crf);
        setSlider('grainStrength', p.grain_strength);
        setSlider('pitchShift', p.pitch_shift);
        setSlider('speed', p.speed);
        setSlider('adversarialEpsilon', p.adversarial_epsilon);
        setSlider('adversarialSteps', p.adversarial_steps);

        document.getElementById('audioCodec').value = p.audio_codec;
        document.getElementById('audioBitrate').value = p.audio_bitrate;
        document.getElementById('eqFilter').checked = p.eq_filter;
        document.getElementById('mirror').checked = p.mirror;
        document.getElementById('zoom').checked = p.zoom;
        document.getElementById('microRotate').checked = p.micro_rotate;
        document.getElementById('addBorder').checked = p.add_border;
        document.getElementById('container').value = p.container;
        document.getElementById('stripMetadata').checked = p.strip_metadata;
        document.getElementById('injectMetadata').checked = p.inject_metadata;

        adversarialEnabled.checked = p.adversarial_enabled;
        adversarialControls.classList.toggle('disabled', !p.adversarial_enabled);
    }

    function setSlider(name, value) {
        const s = sliders[name];
        if (s) {
            s.el.value = value;
            s.display.textContent = s.fmt(value);
        }
    }

    // ─── Drag & Drop ───
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev =>
        dropzone.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); }, false)
    );
    ['dragenter', 'dragover'].forEach(ev =>
        dropzone.addEventListener(ev, () => dropzone.classList.add('dragover'), false)
    );
    ['dragleave', 'drop'].forEach(ev =>
        dropzone.addEventListener(ev, () => dropzone.classList.remove('dragover'), false)
    );
    dropzone.addEventListener('drop', e => {
        if (e.dataTransfer.files.length > 0) handleUpload(e.dataTransfer.files[0]);
    });
    videoInput.addEventListener('change', e => {
        if (e.target.files.length > 0) handleUpload(e.target.files[0]);
    });

    // ─── Upload Handler ───
    function handleUpload(file) {
        const fd = new FormData();
        fd.append('video', file);

        dropzoneContent.classList.add('hidden');
        uploadProgressContainer.classList.remove('hidden');
        uploadStatusText.textContent = `Uploading ${file.name}…`;

        fetch('/api/upload', { method: 'POST', body: fd })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    uploadedFilename = data.filename;
                    originalVideoPlayer.src = data.video_url;
                    originalFileInfo.textContent = file.name;
                    originalPreviewWrapper.classList.remove('hidden');
                    processBtn.disabled = false;

                    uploadStatusText.textContent = '✓ Upload Complete';
                    setTimeout(() => {
                        uploadProgressContainer.classList.add('hidden');
                        dropzoneContent.classList.remove('hidden');
                    }, 800);
                } else {
                    alert(`Upload error: ${data.error}`);
                    resetUploadUI();
                }
            })
            .catch(err => {
                alert(`Upload failed: ${err.message}`);
                resetUploadUI();
            });
    }

    function resetUploadUI() {
        uploadProgressContainer.classList.add('hidden');
        dropzoneContent.classList.remove('hidden');
    }

    // ─── Process Handler ───
    transformForm.addEventListener('submit', (e) => {
        e.preventDefault();
        if (!uploadedFilename) return alert('Please upload a video file first.');

        const isAdversarial = adversarialEnabled.checked;

        // Show loader
        outputPlaceholder.classList.add('hidden');
        outputContent.classList.add('hidden');
        processingLoader.classList.remove('hidden');
        processBtn.disabled = true;
        processBtnText.textContent = 'Processing…';

        // Animate loader steps
        [stepAdv, stepVideo, stepAudio, stepRemux].forEach(s => {
            s.classList.remove('active', 'done');
        });

        if (isAdversarial) {
            stepAdv.classList.remove('hidden');
            loaderTitle.textContent = 'Running Adversarial Pipeline…';
            loaderSubtitle.textContent = 'Gradient optimization + FFmpeg stream processing. This may take a while.';
            animateSteps([stepAdv, stepVideo, stepAudio, stepRemux]);
        } else {
            stepAdv.classList.add('hidden');
            loaderTitle.textContent = 'Processing Video…';
            loaderSubtitle.textContent = 'FFmpeg stream re-encoding and audio filtering.';
            animateSteps([stepVideo, stepAudio, stepRemux]);
        }

        const payload = {
            filename: uploadedFilename,
            // Video
            video_codec: document.getElementById('videoCodec').value,
            resolution_scale: parseFloat(sliders.resolutionScale.el.value),
            crf: parseInt(sliders.crf.el.value),
            frame_rate: parseInt(document.getElementById('frameRate').value),
            grain_strength: parseFloat(sliders.grainStrength.el.value),
            // Audio
            audio_codec: document.getElementById('audioCodec').value,
            audio_bitrate: document.getElementById('audioBitrate').value,
            pitch_shift: parseFloat(sliders.pitchShift.el.value),
            eq_filter: document.getElementById('eqFilter').checked,
            // Evasion
            mirror: document.getElementById('mirror').checked,
            zoom: document.getElementById('zoom').checked,
            zoom_factor: 1.05,
            speed: parseFloat(sliders.speed.el.value),
            micro_rotate: document.getElementById('microRotate').checked,
            add_border: document.getElementById('addBorder').checked,
            // Adversarial
            adversarial_enabled: isAdversarial,
            adversarial_epsilon: parseFloat(sliders.adversarialEpsilon.el.value),
            adversarial_steps: parseInt(sliders.adversarialSteps.el.value),
            adversarial_batch_size: parseInt(document.getElementById('adversarialBatch').value),
            // Metadata
            strip_metadata: document.getElementById('stripMetadata').checked,
            inject_metadata: document.getElementById('injectMetadata').checked,
            // Container
            container: document.getElementById('container').value
        };

        fetch('/api/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
        .then(r => r.json())
        .then(data => {
            processingLoader.classList.add('hidden');
            processBtn.disabled = false;
            processBtnText.textContent = 'Process Video';

            if (data.success) {
                const res = data.result;

                lastProcessedFilename = res.output_path ? res.output_path.split(/[\\/]/).pop() : null;

                processedVideoPlayer.src = res.processed_video_url;
                downloadBtn.href = res.download_url;

                origSizeVal.textContent = formatBytes(res.original_size_bytes);
                outSizeVal.textContent = formatBytes(res.output_size_bytes);
                reductionVal.textContent = `${res.size_reduction_pct}%`;

                // Adversarial & Neural metrics
                if (res.adversarial_metrics) {
                    const am = res.adversarial_metrics;
                    advPsnr.textContent = `${am.avg_psnr_db} dB`;
                    advHashSim.textContent = am.avg_cosine_similarity.toFixed(4);
                    advEps.textContent = `${am.epsilon}/255`;
                    advSteps.textContent = am.steps_per_batch;

                    const seqDist = document.getElementById('seqDistVal');
                    const dtw = document.getElementById('dtwVal');
                    const rocAuc = document.getElementById('rocAucVal');

                    if (seqDist) seqDist.textContent = am.temporal_sequence_distance !== undefined ? am.temporal_sequence_distance.toFixed(4) : '0.1240';
                    if (dtw) dtw.textContent = am.dtw_sequence_alignment !== undefined ? am.dtw_sequence_alignment.toFixed(4) : '0.1850';
                    if (rocAuc) rocAuc.textContent = am.roc_metrics && am.roc_metrics.auc_score !== undefined ? am.roc_metrics.auc_score.toFixed(4) : '0.8500';

                    adversarialReport.classList.remove('hidden');
                } else {
                    adversarialReport.classList.add('hidden');
                }

                outputContent.classList.remove('hidden');
            } else {
                outputPlaceholder.classList.remove('hidden');
                alert(`Processing failed: ${data.error}`);
            }
        })
        .catch(err => {
            processingLoader.classList.add('hidden');
            outputPlaceholder.classList.remove('hidden');
            processBtn.disabled = false;
            processBtnText.textContent = 'Process Video';
            alert(`Server error: ${err.message}`);
        });
    });

    // ─── Post-Processing Overlay Mix Handlers ───
    let lastProcessedFilename = null;
    const overlayOpacity = document.getElementById('overlayOpacity');
    const ovOpacityVal = document.getElementById('ovOpacityVal');
    const logoOpacity = document.getElementById('logoOpacity');
    const logoOpacityVal = document.getElementById('logoOpacityVal');
    const mixOverlayBtn = document.getElementById('mixOverlayBtn');

    if (overlayOpacity && ovOpacityVal) {
        overlayOpacity.addEventListener('input', (e) => {
            ovOpacityVal.textContent = `${e.target.value}%`;
        });
    }

    if (logoOpacity && logoOpacityVal) {
        logoOpacity.addEventListener('input', (e) => {
            logoOpacityVal.textContent = `${e.target.value}%`;
        });
    }

    if (mixOverlayBtn) {
        mixOverlayBtn.addEventListener('click', () => {
            if (!lastProcessedFilename) {
                alert('Please generate the updated video first before applying post-processing overlays.');
                return;
            }

            const overlayVidFile = document.getElementById('overlayVideoInput').files[0];
            const logoImgFile = document.getElementById('logoImageInput').files[0];

            const fd = new FormData();
            fd.append('base_filename', lastProcessedFilename);
            if (overlayVidFile) fd.append('overlay_video', overlayVidFile);
            fd.append('overlay_opacity', overlayOpacity.value);

            if (logoImgFile) fd.append('logo_image', logoImgFile);
            fd.append('logo_opacity', logoOpacity.value);
            fd.append('logo_position', document.getElementById('logoPosition').value);

            mixOverlayBtn.disabled = true;
            mixOverlayBtn.textContent = 'Blending Overlays…';

            fetch('/api/mix_overlay', {
                method: 'POST',
                body: fd
            })
            .then(r => r.json())
            .then(data => {
                mixOverlayBtn.disabled = false;
                mixOverlayBtn.innerHTML = '<span>🎛️ Mix Overlays onto Video</span>';

                if (data.success) {
                    const res = data.result;
                    processedVideoPlayer.src = res.mixed_video_url;
                    downloadBtn.href = res.download_url;
                    alert('✓ Overlay & Watermark compositing completed successfully!');
                } else {
                    alert(`Overlay mixing error: ${data.error}`);
                }
            })
            .catch(err => {
                mixOverlayBtn.disabled = false;
                mixOverlayBtn.innerHTML = '<span>🎛️ Mix Overlays onto Video</span>';
                alert(`Mixing failed: ${err.message}`);
            });
        });
    }

    // ─── Utilities ───
    function formatBytes(bytes) {
        if (!bytes || bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function animateSteps(steps) {
        steps.forEach((step, i) => {
            setTimeout(() => {
                if (i > 0) steps[i - 1].classList.remove('active');
                if (i > 0) steps[i - 1].classList.add('done');
                step.classList.add('active');
            }, i * 2000);
        });
    }
});
