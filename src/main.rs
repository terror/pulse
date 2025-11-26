use {
  anyhow::{bail, Context},
  cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, FromSample, Sample, SizedSample, StreamConfig,
  },
  crossbeam_channel::{unbounded, Receiver, Sender},
  crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{
      disable_raw_mode, enable_raw_mode, EnterAlternateScreen,
      LeaveAlternateScreen,
    },
  },
  num_traits::ToPrimitive,
  ratatui::{
    backend::CrosstermBackend, buffer::Buffer, layout::Rect, style::Color,
    widgets::Widget, Terminal,
  },
  realfft::RealFftPlanner,
  std::{
    collections::VecDeque,
    io,
    sync::{
      atomic::{AtomicBool, Ordering},
      Arc,
    },
    thread,
    time::{Duration, Instant},
  },
};

const BAR_CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
const BAR_COUNT: usize = 32;
const FFT_SIZE: usize = 2048;

struct SpectrumWidget<'a> {
  bars: &'a [f32],
  peaks: &'a [f32],
}

impl<'a> SpectrumWidget<'a> {
  fn new(bars: &'a [f32], peaks: &'a [f32]) -> Self {
    Self { bars, peaks }
  }
}

impl Widget for SpectrumWidget<'_> {
  fn render(self, area: Rect, buf: &mut Buffer) {
    if area.width < 4 || area.height < 2 {
      return;
    }

    let chart_height = f32::from(area.height).max(1.0);

    let bar_width = 2u16;
    let gap = 1u16;

    let total_bar_width = bar_width + gap;

    let max_bars = usize::from(area.width / total_bar_width);
    let bars_to_show = max_bars.min(self.bars.len());
    let bars_to_show_u16 = u16::try_from(bars_to_show).unwrap_or(u16::MAX);

    let start_x =
      area.x + (area.width - (bars_to_show_u16 * total_bar_width)) / 2;

    for (i, (&value, &peak)) in self
      .bars
      .iter()
      .zip(self.peaks.iter())
      .take(bars_to_show)
      .enumerate()
    {
      let i_u16 = u16::try_from(i).unwrap_or(u16::MAX);

      let x = start_x + (i_u16 * total_bar_width);

      let normalized = (value / 6.0).clamp(0.0, 1.0);
      let peak_normalized = (peak / 6.0).clamp(0.0, 1.0);

      let bar_height =
        clamp_round_to_u16(normalized * chart_height).min(area.height);

      let peak_height =
        clamp_round_to_u16(peak_normalized * chart_height).min(area.height);

      let peak_y = area.y + area.height - 1 - peak_height;

      let color = bar_color(i, normalized);

      for row in 0..bar_height {
        let y = area.y + area.height - 1 - row;

        if y >= area.y {
          let row_f32 = f32::from(row);

          let char_idx_f32 = (row_f32 / chart_height * 7.0).clamp(0.0, 7.0);
          let char_idx = clamp_round_to_usize(char_idx_f32, 0, 7);

          let c = BAR_CHARS[char_idx];
          let fade = (row_f32 / f32::from(bar_height)).powf(0.7);

          if let Color::Rgb(cr, cg, cb) = color {
            let r_f32 = f32::from(cr) * (0.4 + 0.6 * fade);
            let g_f32 = f32::from(cg) * (0.4 + 0.6 * fade);
            let b_f32 = f32::from(cb) * (0.4 + 0.6 * fade);

            let dimmed = Color::Rgb(
              clamp_round_to_u8(r_f32),
              clamp_round_to_u8(g_f32),
              clamp_round_to_u8(b_f32),
            );

            for dx in 0..bar_width {
              if x + dx < area.x + area.width {
                buf[(x + dx, y)].set_char(c).set_fg(dimmed);
              }
            }
          }
        }
      }

      if peak_y >= area.y && peak_y < area.y + area.height {
        let peak_color = Color::Rgb(255, 255, 255);

        for dx in 0..bar_width {
          if x + dx < area.x + area.width {
            buf[(x + dx, peak_y)].set_char('━').set_fg(peak_color);
          }
        }
      }
    }
  }
}

fn clamp_round_to_u8(value: f32) -> u8 {
  value
    .is_finite()
    .then(|| value.clamp(0.0, f32::from(u8::MAX)).round())
    .and_then(|rounded| rounded.to_u8())
    .unwrap_or_default()
}

fn clamp_round_to_u16(value: f32) -> u16 {
  value
    .is_finite()
    .then(|| value.clamp(0.0, f32::from(u16::MAX)).round())
    .and_then(|rounded| rounded.to_u16())
    .unwrap_or_default()
}

fn clamp_round_to_usize(value: f32, min: usize, max: usize) -> usize {
  if max < min || !value.is_finite() {
    return min;
  }

  let min_clamped = min.min(u32::MAX as usize);
  let max_clamped = max.min(u32::MAX as usize);

  let (min_f32, max_f32) = match (min_clamped.to_f32(), max_clamped.to_f32()) {
    (Some(min_f32), Some(max_f32)) if max_f32 > min_f32 => (min_f32, max_f32),
    _ => return min_clamped,
  };

  value
    .clamp(min_f32, max_f32)
    .round()
    .to_usize()
    .unwrap_or(min_clamped)
}

fn bar_color(bar_index: usize, intensity: f32) -> Color {
  let bar_index_u16 =
    u16::try_from(bar_index.min(u16::MAX as usize)).unwrap_or(u16::MAX);

  let bar_count_u16 =
    u16::try_from(BAR_COUNT.min(u16::MAX as usize)).unwrap_or(u16::MAX);

  let position = f32::from(bar_index_u16) / f32::from(bar_count_u16);

  let boost = (intensity * 0.3).min(0.3);

  let (r, g, b) = if position < 0.33 {
    let t = position / 0.33;
    (
      clamp_round_to_u8(180.0 + t * 60.0 + boost * 50.0),
      clamp_round_to_u8(50.0 + t * 30.0),
      clamp_round_to_u8(220.0 - t * 40.0 + boost * 35.0),
    )
  } else if position < 0.66 {
    let t = (position - 0.33) / 0.33;
    (
      clamp_round_to_u8(50.0 + t * 30.0),
      clamp_round_to_u8(200.0 + boost * 55.0),
      clamp_round_to_u8(220.0 - t * 60.0 + boost * 35.0),
    )
  } else {
    let t = (position - 0.66) / 0.34;
    (
      clamp_round_to_u8(240.0 + boost * 15.0),
      clamp_round_to_u8(100.0 + t * 50.0 + boost * 50.0),
      clamp_round_to_u8(120.0 + t * 60.0),
    )
  };

  Color::Rgb(r, g, b)
}

fn audio_loop(
  running: Arc<AtomicBool>,
  tx: Sender<Vec<f32>>,
) -> anyhow::Result<()> {
  let host = cpal::default_host();

  let device = find_blackhole_input_device(&host).or_else(|_| {
    host
      .default_input_device()
      .context("No default input device found")
  })?;

  let config = device.default_input_config()?;
  let sample_format = config.sample_format();
  let config: StreamConfig = config.into();

  let err_fn = |err| eprintln!("Stream error: {err}");

  match sample_format {
    cpal::SampleFormat::F32 => {
      build_stream::<f32>(&device, &config, err_fn, running, tx)?;
    }
    cpal::SampleFormat::I16 => {
      build_stream::<i16>(&device, &config, err_fn, running, tx)?;
    }
    cpal::SampleFormat::U16 => {
      build_stream::<u16>(&device, &config, err_fn, running, tx)?;
    }
    _ => unimplemented!("Unsupported sample format"),
  }

  Ok(())
}

fn find_blackhole_input_device(host: &cpal::Host) -> anyhow::Result<Device> {
  for device in host.input_devices()? {
    let name = device.name()?;

    if name.contains("BlackHole") {
      return Ok(device);
    }
  }

  bail!("BlackHole device not found");
}

fn build_stream<T>(
  device: &Device,
  config: &StreamConfig,
  err_fn: impl Fn(cpal::StreamError) + Send + 'static,
  running: Arc<AtomicBool>,
  tx: Sender<Vec<f32>>,
) -> anyhow::Result<()>
where
  T: Sample + SizedSample,
  f32: FromSample<T>,
{
  let channels = config.channels as usize;

  let mut planner = RealFftPlanner::<f32>::new();

  let r2c = planner.plan_fft_forward(FFT_SIZE);

  let mut input_buf = r2c.make_input_vec();
  let mut output_buf = r2c.make_output_vec();

  let mut samples = VecDeque::with_capacity(FFT_SIZE);

  for _ in 0..FFT_SIZE {
    samples.push_back(0.0);
  }

  let window: Vec<f32> = (0..FFT_SIZE)
    .map(|i| {
      let i_u16 = u16::try_from(i.min(u16::MAX as usize)).unwrap_or(u16::MAX);
      let fft_size_u16 =
        u16::try_from(FFT_SIZE.min(u16::MAX as usize)).unwrap_or(u16::MAX);
      let i_f32 = f32::from(i_u16);
      let fft_size_f32 = f32::from(fft_size_u16);
      0.5
        * (1.0
          - (2.0 * std::f32::consts::PI * i_f32 / (fft_size_f32 - 1.0)).cos())
    })
    .collect();

  let mut last_send = Instant::now();

  let send_interval = Duration::from_millis(20);

  let running_inner = running.clone();

  let stream = device.build_input_stream(
    config,
    move |data: &[T], _: &cpal::InputCallbackInfo| {
      if !running_inner.load(Ordering::Relaxed) {
        return;
      }

      for frame in data.chunks(channels) {
        let mut sum = 0.0f32;

        for &s in frame {
          sum += s.to_sample::<f32>();
        }

        let channels_u16 =
          u16::try_from(channels.min(u16::MAX as usize)).unwrap_or(u16::MAX);
        let mono = sum / f32::from(channels_u16);

        samples.pop_front();
        samples.push_back(mono);
      }

      if last_send.elapsed() >= send_interval {
        for (i, &s) in samples.iter().enumerate() {
          input_buf[i] = s * window[i];
        }

        if let Ok(()) = r2c.process(&mut input_buf, &mut output_buf) {
          let spectrum_len = output_buf.len();

          let mut bars = vec![0.0; BAR_COUNT];

          let min_freq = 20.0f32;
          let max_freq = 16000.0f32;

          let sample_rate = 44100.0f32;

          let fft_size_u16 =
            u16::try_from(FFT_SIZE.min(u16::MAX as usize)).unwrap_or(u16::MAX);

          let bin_freq = sample_rate / f32::from(fft_size_u16);

          for (i, bar) in bars.iter_mut().enumerate().take(BAR_COUNT) {
            let i_u16 =
              u16::try_from(i.min(u16::MAX as usize)).unwrap_or(u16::MAX);

            let bar_count_u16 = u16::try_from(BAR_COUNT.min(u16::MAX as usize))
              .unwrap_or(u16::MAX);

            let t0 = f32::from(i_u16) / f32::from(bar_count_u16);

            let i_plus_one_u16 =
              u16::try_from((i + 1).min(u16::MAX as usize)).unwrap_or(u16::MAX);

            let t1 = f32::from(i_plus_one_u16) / f32::from(bar_count_u16);

            let freq0 = min_freq * (max_freq / min_freq).powf(t0);
            let freq1 = min_freq * (max_freq / min_freq).powf(t1);

            let bin0_f32 = (freq0 / bin_freq).max(1.0);

            let spectrum_len_minus_one_u16 =
              u16::try_from((spectrum_len - 1).min(u16::MAX as usize))
                .unwrap_or(u16::MAX);

            let spectrum_len_f32 = f32::from(spectrum_len_minus_one_u16);

            let last_bin = usize::from(spectrum_len_minus_one_u16);

            let bin0 = clamp_round_to_usize(bin0_f32, 1, last_bin.max(1));
            let bin1_f32 = (freq1 / bin_freq).min(spectrum_len_f32);
            let bin1 = clamp_round_to_usize(bin1_f32, 0, last_bin);

            let mut max_mag = 0.0f32;

            for val in output_buf
              .iter()
              .take(bin1 + 1)
              .skip(bin0)
              .map(|x| x.norm())
            {
              if val > max_mag {
                max_mag = val;
              }
            }

            *bar = (max_mag * 10.0).ln_1p();
          }

          let _ = tx.send(bars);
        }

        last_send = Instant::now();
      }
    },
    err_fn,
    None,
  )?;

  stream.play()?;

  while running.load(Ordering::Relaxed) {
    thread::sleep(Duration::from_millis(100));
  }

  Ok(())
}

fn run(running: Arc<AtomicBool>, rx: Receiver<Vec<f32>>) -> anyhow::Result<()> {
  enable_raw_mode()?;

  let mut stdout = io::stdout();
  execute!(stdout, EnterAlternateScreen)?;

  let backend = CrosstermBackend::new(stdout);

  let mut terminal = Terminal::new(backend)?;

  let mut bars = vec![0.0; BAR_COUNT];
  let mut smooth_bars = vec![0.0; BAR_COUNT];
  let mut peaks = vec![0.0; BAR_COUNT];

  loop {
    if event::poll(Duration::from_millis(10))? {
      if let Event::Key(key) = event::read()? {
        if key.code == KeyCode::Char('q') || key.code == KeyCode::Esc {
          running.store(false, Ordering::Relaxed);
          break;
        }
      }
    }

    while let Ok(new_bars) = rx.try_recv() {
      bars = new_bars;
    }

    for i in 0..BAR_COUNT {
      if bars[i] > smooth_bars[i] {
        smooth_bars[i] = 0.5 * smooth_bars[i] + 0.5 * bars[i];
      } else {
        smooth_bars[i] = 0.85 * smooth_bars[i] + 0.15 * bars[i];
      }

      if smooth_bars[i] > peaks[i] {
        peaks[i] = smooth_bars[i];
      } else {
        peaks[i] *= 0.97;
      }
    }

    terminal.draw(|f| {
      let area = f.area();
      let widget = SpectrumWidget::new(&smooth_bars, &peaks);
      f.render_widget(widget, area);
    })?;
  }

  disable_raw_mode()?;
  execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
  terminal.show_cursor()?;

  Ok(())
}

fn main() -> anyhow::Result<()> {
  let running = Arc::new(AtomicBool::new(true));

  let (tx, rx) = unbounded::<Vec<f32>>();

  {
    let running = running.clone();

    thread::spawn(move || {
      if let Err(e) = audio_loop(running, tx) {
        eprintln!("Audio thread error: {e:?}");
      }
    });
  }

  run(running, rx)?;

  Ok(())
}
