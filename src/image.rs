use std::{
    collections::HashMap,
    io::Cursor,
    ops::{Add, Div, Mul, Sub},
};

use image::{io::Reader, DynamicImage};
use wasm_bindgen::{Clamped, JsCast};
use web_sys::{
    CanvasRenderingContext2d, HtmlCanvasElement, HtmlInputElement, HtmlSelectElement, ImageData,
};
use yew::prelude::*;

#[derive(Hash, PartialEq, Eq)]
pub enum ColorComponent {
    Red,
    Green,
    Blue,
    Alpha,
}

pub enum Msg {
    ApplyOperation,
    ValueChanged(Event),
    BrightnessChanged(Event),
    ToGrayscaleAvg,
    ToGrayscaleAvgWeighted,
    FilterSmooth,
    FilterMedian,
    FilterEdgeDetection,
    FilterSharpen,
    FilterGaussianBlur
}

#[derive(Properties, PartialEq)]
pub struct Props {
    pub image_data: Vec<u8>,
}

pub struct Image {
    bitmap_data: Vec<u8>,
    width: u32,
    height: u32,

    canvas_ref: NodeRef,
    canvas_ctx: Option<CanvasRenderingContext2d>,
    color_select_ref: NodeRef,
    operation_select_ref: NodeRef,
    input_value: f32,
    brigthness_scale: f32
}

impl Image {
    pub fn new_with_data(data: Vec<u8>) -> Self {
        let image = Self::decode_data(data);

        Self {
            bitmap_data: image.to_rgba8().into_vec(),
            width: image.width(),
            height: image.height(),

            canvas_ref: NodeRef::default(),
            canvas_ctx: None,
            color_select_ref: NodeRef::default(),
            operation_select_ref: NodeRef::default(),
            input_value: 0.0,
            brigthness_scale: 0.0
        }
    }

    pub fn apply_point_fn(
        &mut self,
        component: ColorComponent,
        value: f32,
        func: &dyn Fn(f32, f32) -> f32,
    ) {
        if value == 0.0 {
            return;
        }

        let mut index = 0;
        let offset = match component {
            ColorComponent::Red => 0,
            ColorComponent::Green => 1,
            ColorComponent::Blue => 2,
            ColorComponent::Alpha => 3,
        };

        while index < self.bitmap_data.len() {
            let color = self.bitmap_data[index + offset] as f32;
            let new_color = func(color, value);

            self.bitmap_data[index + offset] = new_color as u8;

            index += 4;
        }
    }

    pub fn change_brightness(&mut self, brightness: f32) {
        let brightness = brightness / 2.0;
        for i in 0..self.bitmap_data.len() {
            if i % 4 == 3 {
                continue;
            }

            let norm_val = self.bitmap_data[i] as f32 / 255.0;
            let new_val = if brightness < 0.0 {
                norm_val * (1.0 + brightness)
            } else {
                norm_val + brightness * (1.0 - norm_val)
            };

            self.bitmap_data[i] = (new_val * 255.0) as u8;
        }
    }

    pub fn to_grayscale_avg(&mut self) {
        let mut index = 0;

        while index < self.bitmap_data.len() {
            let red = self.bitmap_data[index] as f32;
            let green = self.bitmap_data[index + 1] as f32;
            let blue = self.bitmap_data[index + 2] as f32;

            let avg = (red + green + blue) / 3.0;

            self.bitmap_data[index] = avg as u8;
            self.bitmap_data[index + 1] = avg as u8;
            self.bitmap_data[index + 2] = avg as u8;

            index += 4;
        }
    }

    pub fn to_grayscale_avg_weighted(&mut self) {
        let mut index = 0;

        while index < self.bitmap_data.len() {
            let red = self.bitmap_data[index] as f32;
            let green = self.bitmap_data[index + 1] as f32;
            let blue = self.bitmap_data[index + 2] as f32;

            let avg = (red * 0.2126 + green * 0.7152 + blue * 0.0722) as u8;

            self.bitmap_data[index] = avg;
            self.bitmap_data[index + 1] = avg;
            self.bitmap_data[index + 2] = avg;

            index += 4;
        }
    }

    pub fn filter_smooth(&mut self) {
        let mut index = 0;
        let mut new_bitmap_data = self.bitmap_data.clone();

        while index < self.bitmap_data.len() {
            let mut red = 0;
            let mut green = 0;
            let mut blue = 0;

            for i in 0..9 {
                let x = i % 3;
                let y = i / 3;

                let pixel_index = index + (x - 1) * 4 + (y - 1) * self.width as usize * 4;

                if pixel_index < 0 || pixel_index >= self.bitmap_data.len() {
                    continue;
                }

                red += self.bitmap_data[pixel_index] as usize;
                green += self.bitmap_data[pixel_index + 1] as usize;
                blue += self.bitmap_data[pixel_index + 2] as usize;
            }

            new_bitmap_data[index] = (red / 9) as u8;
            new_bitmap_data[index + 1] = (green / 9) as u8;
            new_bitmap_data[index + 2] = (blue / 9) as u8;

            index += 4;
        }

        self.bitmap_data = new_bitmap_data;
    }

    pub fn filter_median(&mut self) {
        let mut index = 0;
        let mut new_bitmap_data = self.bitmap_data.clone();

        while index < self.bitmap_data.len() {
            let mut red = [0; 9];
            let mut green = [0; 9];
            let mut blue = [0; 9];

            for i in 0..9 {
                let x = i % 3;
                let y = i / 3;

                let pixel_index = index + (x - 1) * 4 + (y - 1) * self.width as usize * 4;

                if pixel_index < 0 || pixel_index >= self.bitmap_data.len() {
                    continue;
                }

                red[i] = self.bitmap_data[pixel_index] as u32;
                green[i] = self.bitmap_data[pixel_index + 1] as u32;
                blue[i] = self.bitmap_data[pixel_index + 2] as u32;
            }

            red.sort();
            green.sort();
            blue.sort();

            new_bitmap_data[index] = red[4] as u8;
            new_bitmap_data[index + 1] = green[4] as u8;
            new_bitmap_data[index + 2] = blue[4] as u8;

            index += 4;
        }

        self.bitmap_data = new_bitmap_data;
    }

    pub fn filter_sobel(&mut self) {
        let mut index = 0;
        let mut new_bitmap_data = self.bitmap_data.clone();

        while index < self.bitmap_data.len() {
            let mut red_x = 0;
            let mut green_x = 0;
            let mut blue_x = 0;

            let mut red_y = 0;
            let mut green_y = 0;
            let mut blue_y = 0;

            for i in 0..9 {
                let x = i % 3;
                let y = i / 3;

                let pixel_index = index + (x - 1) * 4 + (y - 1) * self.width as usize * 4;

                if pixel_index < 0 || pixel_index >= self.bitmap_data.len() {
                    continue;
                }

                let red = self.bitmap_data[pixel_index] as i32;
                let green = self.bitmap_data[pixel_index + 1] as i32;
                let blue = self.bitmap_data[pixel_index + 2] as i32;

                let x_weight = match x {
                    0 => -1,
                    1 => 0,
                    2 => 1,
                    _ => unreachable!(),
                };

                let y_weight = match y {
                    0 => -1,
                    1 => 0,
                    2 => 1,
                    _ => unreachable!(),
                };

                red_x += red * x_weight;
                green_x += green * x_weight;
                blue_x += blue * x_weight;

                red_y += red * y_weight;
                green_y += green * y_weight;
                blue_y += blue * y_weight;
            }

            let red = ((red_x * red_x + red_y * red_y) as f32).sqrt() as u8;
            let green = ((green_x * green_x + green_y * green_y) as f32).sqrt() as u8;
            let blue = ((blue_x * blue_x + blue_y * blue_y) as f32).sqrt() as u8;

            new_bitmap_data[index] = red;
            new_bitmap_data[index + 1] = green;
            new_bitmap_data[index + 2] = blue;

            index += 4;
        }

        self.bitmap_data = new_bitmap_data;
    }

    pub fn filter_highpass_sharpen(&mut self) {
        let mut highpass_data = self.bitmap_data.clone();

        let mut index = 0;
        while index < self.bitmap_data.len() {
            let mut red: f32 = 0.0;
            let mut green: f32 = 0.0;
            let mut blue: f32 = 0.0;

            for i in 0..9 {
                let x = i % 3;
                let y = i / 3;

                let pixel_index = index + (x - 1) * 4 + (y - 1) * self.width as usize * 4;

                if pixel_index < 0 || pixel_index >= self.bitmap_data.len() {
                    continue;
                }

                let weight: f32 = match (x, y) {
                    (1, 1) => 8.0 / 9.0,
                    _ => -1.0 / 9.0,
                };

                red += f32::from(self.bitmap_data[pixel_index]) * weight;
                green += f32::from(self.bitmap_data[pixel_index + 1]) * weight;
                blue += f32::from(self.bitmap_data[pixel_index + 2]) * weight;
            }

            highpass_data[index] = red as u8;
            highpass_data[index + 1] = green as u8;
            highpass_data[index + 2] = blue as u8;

            index += 4;
        }

        index = 0;

        while index < self.bitmap_data.len() {
            let red = self.bitmap_data[index].saturating_add(highpass_data[index]);
            let green = self.bitmap_data[index + 1].saturating_add(highpass_data[index + 1]);
            let blue = self.bitmap_data[index + 2].saturating_add(highpass_data[index + 2]);

            self.bitmap_data[index] = red;
            self.bitmap_data[index + 1] = green;
            self.bitmap_data[index + 2] = blue;

            index += 4;
        }
    }

    pub fn filter_gaussian_blur(&mut self) {
        let mut index = 0;
        let mut new_bitmap_data = self.bitmap_data.clone();

        while index < self.bitmap_data.len() {
            let mut red = 0;
            let mut green = 0;
            let mut blue = 0;

            for i in 0..9 {
                let x = i % 3;
                let y = i / 3;

                let pixel_index = index + (x - 1) * 4 + (y - 1) * self.width as usize * 4;

                if pixel_index < 0 || pixel_index >= self.bitmap_data.len() {
                    continue;
                }

                let weight = match (x, y) {
                    (0, 0) => 1,
                    (1, 0) => 2,
                    (2, 0) => 1,
                    (0, 1) => 2,
                    (1, 1) => 4,
                    (2, 1) => 2,
                    (0, 2) => 1,
                    (1, 2) => 2,
                    (2, 2) => 1,
                    _ => unreachable!(),
                };

                red += self.bitmap_data[pixel_index] as i32 * weight;
                green += self.bitmap_data[pixel_index + 1] as i32 * weight;
                blue += self.bitmap_data[pixel_index + 2] as i32 * weight;
            }

            new_bitmap_data[index] = (red / 16) as u8;
            new_bitmap_data[index + 1] = (green / 16) as u8;
            new_bitmap_data[index + 2] = (blue / 16) as u8;

            index += 4;
        }

        self.bitmap_data = new_bitmap_data;
    }

    fn rgb_to_hsl(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let r = r / 255.0;
        let g = g / 255.0;
        let b = b / 255.0;

        let max = r.max(g.max(b));
        let min = r.min(g.min(b));

        let mut h = 0.0;
        let mut s = 0.0;
        let l = (max + min) / 2.0;

        if max != min {
            let d = max - min;
            s = if l > 0.5 {
                d / (2.0 - max - min)
            } else {
                d / (max + min)
            };

            if max == r {
                h = (g - b) / d + if g < b { 6.0 } else { 0.0 };
            } else if max == g {
                h = (b - r) / d + 2.0;
            } else if max == b {
                h = (r - g) / d + 4.0;
            }

            h /= 6.0;
        }

        (h, s, l)
    }

    fn update(&mut self, data: Vec<u8>) {
        let image = Self::decode_data(data);

        self.bitmap_data = image.to_rgba8().into_vec();
        self.width = image.width();
        self.height = image.height();
    }

    fn decode_data(data: Vec<u8>) -> DynamicImage {
        let reader = Reader::new(Cursor::new(&data[..]))
            .with_guessed_format()
            .expect("Couldn't guess file format.");

        reader.decode().expect("Unable to decode image.")
    }
}

impl Component for Image {
    type Message = Msg;
    type Properties = Props;

    fn create(ctx: &Context<Self>) -> Self {
        Self::new_with_data(ctx.props().image_data.clone())
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        html! {
            <>
                <div>
                    <div>
                        <label>{"Color component"}</label>
                        <select ref={self.color_select_ref.clone()}>
                            <option value="red">{ "Red" }</option>
                            <option value="green">{ "Green" }</option>
                            <option value="blue">{ "Blue" }</option>
                            <option value="alpha">{ "Alpha" }</option>
                        </select>
                        <label>{"Operation"}</label>
                        <select ref={self.operation_select_ref.clone()}>
                            <option value="add">{ "Add" }</option>
                            <option value="subtract">{ "Subtract" }</option>
                            <option value="multiply">{ "Multiply" }</option>
                            <option value="divide">{ "Divide" }</option>
                        </select>
                        <input type="number" min="0" max="255" step="1" value={self.input_value.to_string()}
                            onchange={ctx.link().callback(|event: Event| Msg::ValueChanged(event))} />
                        <input type="button" onclick={ctx.link().callback(|_| Msg::ApplyOperation)} value="Apply" />
                    </div>
                    <div>
                        <label>{"Brightness"}</label>
                        <input type="range" min="-1" max="1" step="0.01" value="0"
                            onchange={ctx.link().callback(|event: Event| Msg::BrightnessChanged(event))} />
                    </div>
                    <div>
                        <input type="button" onclick={ctx.link().callback(|_| Msg::ToGrayscaleAvg)} value="To grayscale (avg)" />
                        <input type="button" onclick={ctx.link().callback(|_| Msg::ToGrayscaleAvgWeighted)} value="To grayscale (avg weighted)" />
                        <input type="button" onclick={ctx.link().callback(|_| Msg::FilterSmooth)} value="Filter (smooth)" />
                        <input type="button" onclick={ctx.link().callback(|_| Msg::FilterMedian)} value="Filter (median)" />
                        <input type="button" onclick={ctx.link().callback(|_| Msg::FilterEdgeDetection)} value="Filter (edge detection)" />
                        <input type="button" onclick={ctx.link().callback(|_| Msg::FilterSharpen)} value="Filter (sharpen)" />
                        <input type="button" onclick={ctx.link().callback(|_| Msg::FilterGaussianBlur)} value="Filter (gaussian blur)" />
                    </div>
                </div>
                <div>
                    <canvas
                        ref={self.canvas_ref.clone()}
                        width={self.width.to_string()}
                        height={self.height.to_string()}
                    />
                </div>
            </>
        }
    }

    fn update(&mut self, _ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::ApplyOperation => {
                let color_select = self.color_select_ref.cast::<HtmlSelectElement>().unwrap();
                let color = match color_select.value().as_str() {
                    "red" => ColorComponent::Red,
                    "green" => ColorComponent::Green,
                    "blue" => ColorComponent::Blue,
                    "alpha" => ColorComponent::Alpha,
                    _ => panic!("Invalid color selection"),
                };

                let op_select = self
                    .operation_select_ref
                    .cast::<HtmlSelectElement>()
                    .unwrap();
                let op: &dyn Fn(f32, f32) -> f32 = match op_select.value().as_str() {
                    "add" => &f32::add,
                    "subtract" => &f32::sub,
                    "multiply" => &f32::mul,
                    "divide" => &f32::div,
                    _ => panic!("Invalid operation selection"),
                };

                self.apply_point_fn(color, self.input_value, op);

                true
            }
            Msg::ValueChanged(event) => {
                let input: HtmlInputElement = event.target().unwrap().dyn_into().unwrap();
                self.input_value = input.value_as_number() as f32;

                true
            }
            Msg::BrightnessChanged(event) => {
                let input: HtmlInputElement = event.target().unwrap().dyn_into().unwrap();
                self.brigthness_scale = input.value_as_number() as f32;
                self.change_brightness(self.brigthness_scale);

                true
            }
            Msg::ToGrayscaleAvg => {
                self.to_grayscale_avg();

                true
            },
            Msg::ToGrayscaleAvgWeighted => {
                self.to_grayscale_avg_weighted();

                true
            },
            Msg::FilterSmooth => {
                self.filter_smooth();

                true
            },
            Msg::FilterMedian => {
                self.filter_median();

                true
            },
            Msg::FilterEdgeDetection => {
                self.filter_sobel();

                true
            },
            Msg::FilterSharpen => {
                self.filter_highpass_sharpen();

                true
            },
            Msg::FilterGaussianBlur => {
                self.filter_gaussian_blur();

                true
            },
        }
    }

    fn changed(&mut self, ctx: &Context<Self>) -> bool {
        let new_data = ctx.props().image_data.clone();
        self.update(new_data);

        true
    }

    fn rendered(&mut self, _ctx: &Context<Self>, first_render: bool) {
        if first_render {
            self.canvas_ctx = Some(
                self.canvas_ref
                    .cast::<HtmlCanvasElement>()
                    .unwrap()
                    .get_context("2d")
                    .unwrap()
                    .unwrap()
                    .dyn_into::<CanvasRenderingContext2d>()
                    .unwrap(),
            );
        }

        let canvas_ctx = self.canvas_ctx.as_ref().unwrap();
        let image_data = ImageData::new_with_u8_clamped_array_and_sh(
            Clamped(&self.bitmap_data),
            self.width,
            self.height,
        )
        .unwrap();

        canvas_ctx.clear_rect(0.0, 0.0, self.width.into(), self.height.into());
        canvas_ctx.set_image_smoothing_enabled(false);
        canvas_ctx
            .put_image_data(&image_data, 0.0, 0.0)
            .expect("Couldn't draw image");
    }
}
