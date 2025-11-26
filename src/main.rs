use {
  anyhow::Context,
  cpal::traits::{DeviceTrait, HostTrait, StreamTrait},
  crossbeam_channel::{unbounded, Receiver},
  crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{
      disable_raw_mode, enable_raw_mode, EnterAlternateScreen,
      LeaveAlternateScreen,
    },
  },
  ratatui::{
    backend::CrosstermBackend,
    layout::Rect,
    style::Color,
    widgets::Widget,
    Terminal,
  },
  realfft::RealFftPlanner,
  std::{
    collections::VecDeque,
    io,
    sync::{
      atomic::{AtomicBool, Ordering},
      Arc,
    },
    time::{Duration, Instant},
  },
};

const BAR_CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
const BAR_COUNT: usize = 32;
const FFT_SIZE: usize = 2048;

fn bar_color(bar_index: usize, intensity: f32) -> Color {
  let position = bar_index as f32 / BAR_COUNT as f32;

  let boost = (intensity * 0.3).min(0.3);

  let (r, g, b) = if position < 0.33 {
    let t = position / 0.33;
    (
      (180.0 + t * 60.0 + boost * 50.0) as u8,
      (50.0 + t * 30.0) as u8,
      (220.0 - t * 40.0 + boost * 35.0) as u8,
    )
  } else if position < 0.66 {
    let t = (position - 0.33) / 0.33;
    (
      (50.0 + t * 30.0) as u8,
      (200.0 + boost * 55.0) as u8,
      (220.0 - t * 60.0 + boost * 35.0) as u8,
    )
  } else {
    let t = (position - 0.66) / 0.34;
    (
      (240.0 + boost * 15.0) as u8,
      (100.0 + t * 50.0 + boost * 50.0) as u8,
      (120.0 + t * 60.0) as u8,
    )
  };

  Color::Rgb(r, g, b)
}

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
  fn render(self, area: Rect, buf: &mut ratatui::buffer::Buffer) {
    if area.width < 4 || area.height < 2 {
      return;
    }

    let chart_height = (area.height as f32).max(1.0);
    let bar_width = 2u16;
    let gap = 1u16;
    let total_bar_width = bar_width + gap;
    let max_bars = (area.width / total_bar_width) as usize;
    let bars_to_show = max_bars.min(self.bars.len());

    let start_x =
      area.x + (area.width - (bars_to_show as u16 * total_bar_width)) / 2;

    for (i, (&value, &peak)) in self
      .bars
      .iter()
      .zip(self.peaks.iter())
      .take(bars_to_show)
      .enumerate()
    {
      let x = start_x + (i as u16 * total_bar_width);
      let normalized = (value / 6.0).clamp(0.0, 1.0);
      let peak_normalized = (peak / 6.0).clamp(0.0, 1.0);
      let bar_height = (normalized * chart_height) as u16;
      let peak_y = area.y + area.height
        - 1
        - (peak_normalized * chart_height) as u16;

      let color = bar_color(i, normalized);

      // Draw the bar from bottom up
      for row in 0..bar_height {
        let y = area.y + area.height - 1 - row;
        if y >= area.y {
          let char_idx =
            ((row as f32 / chart_height * 7.0) as usize).min(7);
          let c = BAR_CHARS[char_idx];
          let fade = (row as f32 / bar_height as f32).powf(0.7);
          if let Color::Rgb(cr, cg, cb) = color {
            let dimmed = Color::Rgb(
              (cr as f32 * (0.4 + 0.6 * fade)) as u8,
              (cg as f32 * (0.4 + 0.6 * fade)) as u8,
              (cb as f32 * (0.4 + 0.6 * fade)) as u8,
            );
            for dx in 0..bar_width {
              if x + dx < area.x + area.width {
                buf[(x + dx, y)].set_char(c).set_fg(dimmed);
              }
            }
          }
        }
      }

      // Draw peak indicator
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

fn audio_loop(
  running: Arc<AtomicBool>,
  tx: crossbeam_channel::Sender<Vec<f32>>,
) -> anyhow::Result<()> {
  let host = cpal::default_host();

  let device = find_blackhole_input_device(&host).or_else(|_| {
    host
      .default_input_device()
      .context("No default input device found")
  })?;


  let config = device.default_input_config()?;
  let sample_format = config.sample_format();
  let config: cpal::StreamConfig = config.into();

  let err_fn = |err| eprintln!("Stream error: {}", err);

  match sample_format {
    cpal::SampleFormat::F32 => {
      build_stream::<f32>(&device, &config, err_fn, running, tx)?
    }
    cpal::SampleFormat::I16 => {
      build_stream::<i16>(&device, &config, err_fn, running, tx)?
    }
    cpal::SampleFormat::U16 => {
      build_stream::<u16>(&device, &config, err_fn, running, tx)?
    }
    _ => unimplemented!("Unsupported sample format"),
  }

  Ok(())
}

fn find_blackhole_input_device(
  host: &cpal::Host,
) -> anyhow::Result<cpal::Device> {
  for device in host.input_devices()? {
    let name = device.name()?;

    if name.contains("BlackHole") {
      return Ok(device);
    }
  }

  anyhow::bail!("BlackHole device not found");
}

fn build_stream<T>(
  device: &cpal::Device,
  config: &cpal::StreamConfig,
  err_fn: impl Fn(cpal::StreamError) + Send + 'static,
  running: Arc<AtomicBool>,
  tx: crossbeam_channel::Sender<Vec<f32>>,
) -> anyhow::Result<()>
where
  T: cpal::Sample + cpal::SizedSample,
  f32: cpal::FromSample<T>,
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
      0.5
        * (1.0
          - (2.0 * std::f32::consts::PI * i as f32 / (FFT_SIZE as f32 - 1.0))
            .cos())
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

        let mono = sum / channels as f32;

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

          // Use logarithmic frequency mapping
          // Map bars to frequency range ~20Hz to ~16kHz
          let min_freq = 20.0f32;
          let max_freq = 16000.0f32;
          let sample_rate = 44100.0f32;
          let bin_freq = sample_rate / FFT_SIZE as f32;

          for i in 0..BAR_COUNT {
            // Logarithmic interpolation between min and max freq
            let t0 = i as f32 / BAR_COUNT as f32;
            let t1 = (i + 1) as f32 / BAR_COUNT as f32;

            let freq0 = min_freq * (max_freq / min_freq).powf(t0);
            let freq1 = min_freq * (max_freq / min_freq).powf(t1);

            let bin0 = ((freq0 / bin_freq) as usize).max(1);
            let bin1 = ((freq1 / bin_freq) as usize).min(spectrum_len - 1);

            let mut max_mag = 0.0f32;

            for j in bin0..=bin1 {
              let val = output_buf[j].norm();

              if val > max_mag {
                max_mag = val;
              }
            }

            bars[i] = (max_mag * 10.0).ln_1p();
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
    std::thread::sleep(Duration::from_millis(100));
  }

  Ok(())
}

fn tick(
  running: Arc<AtomicBool>,
  rx: Receiver<Vec<f32>>,
) -> anyhow::Result<()> {
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
      // Smooth the bars
      if bars[i] > smooth_bars[i] {
        smooth_bars[i] = 0.5 * smooth_bars[i] + 0.5 * bars[i];
      } else {
        smooth_bars[i] = 0.85 * smooth_bars[i] + 0.15 * bars[i];
      }

      // Update peaks with slow falloff
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

    std::thread::spawn(move || {
      if let Err(e) = audio_loop(running, tx) {
        eprintln!("Audio thread error: {:?}", e);
      }
    });
  }

  tick(running, rx)?;

  Ok(())
}
