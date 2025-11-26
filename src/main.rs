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
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    widgets::{BarChart, Block, Borders},
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

const BAR_COUNT: usize = 32;
const FFT_SIZE: usize = 2048;

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

  println!("Capturing from: {}", device.name()?);

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

          let chunk_size = (spectrum_len - 1) / BAR_COUNT;

          for i in 0..BAR_COUNT {
            let start = 1 + i * chunk_size;

            let end = (start + chunk_size).min(spectrum_len);

            let mut max_mag = 0.0f32;

            for j in start..end {
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
        smooth_bars[i] = 0.6 * smooth_bars[i] + 0.4 * bars[i];
      } else {
        smooth_bars[i] = 0.9 * smooth_bars[i] + 0.1 * bars[i];
      }
    }

    let mut bar_data: Vec<(&str, u64)> = Vec::with_capacity(BAR_COUNT);

    for i in 0..BAR_COUNT {
      let height = (smooth_bars[i] * 20.0) as u64;
      bar_data.push(("", height));
    }

    terminal.draw(|f| {
      let area = f.area();

      let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(100)].as_ref())
        .split(area);

      let barchart = BarChart::default()
        .block(
          Block::default()
            .title("Pulse Visualizer (q to quit)")
            .borders(Borders::ALL),
        )
        .data(&bar_data)
        .bar_width(3)
        .bar_gap(1)
        .bar_style(Style::default().fg(Color::Cyan))
        .value_style(Style::default().fg(Color::Black).bg(Color::Cyan));

      f.render_widget(barchart, chunks[0]);
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
