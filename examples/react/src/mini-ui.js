import {Children} from 'react';
import {jsx     } from './jsx.js';
export function Divider() {
  return jsx('hr', null);
}
export function CssBaseline(props) {
  return jsx('div', null, props.children);
}
export function ThemeProvider(props) {
  return jsx('div', null, props.children);
}
export function Container(props) {
  return jsx('div', null, props.children);
}
export function createTheme(settings) {
  return {not: 'implemented'};
}
export function Button(props) {
  return jsx('button', props);
}
export function Grid(props) {
  return jsx('div', props);
}
export function Alert(props) {
  return jsx('div', props);
}
export function Stack(props) {
  const {children} = props;
  const trs = Children.map(children, child => jsx('tr', null, jsx('td', null, child)));
  return jsx('table', null, trs);
}
export function Box(props) {
  return jsx('div', props);
}
export function InputLabel(props) {
  return jsx('div', null, props.children);
}
export function Checkbox(props) {
  return jsx('input', {type: 'checkbox', ...props});
}
export function TextField(TextFieldProps) {
  const {label, disabled, onChange, value, type} = TextFieldProps;
  return (
    jsx('label', null, [
      label,
      jsx('input', {type, disabled, onChange, value}, null)
    ])
  )
}
export function Select(SelectProps) {
  const { value, onChange, children } = SelectProps;
  return jsx('select', {value, onChange}, children);
}
export function MenuItem(MenuItemProps) {
  const { value, disabled, children } = MenuItemProps;
  return jsx('option', {value, disabled}, children);
}
export function FormControlLabel(props) {
  return jsx('label', {}, props.label, ' ', props.control);
}
export function FormControl(props) {
  return jsx('div', null, props.children);
}
