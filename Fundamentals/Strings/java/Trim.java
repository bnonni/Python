import java.util.*;

public class Trim {

	public static String mytrim(String s) {
		ArrayList<String> ch = new ArrayList<String>();
		String str = "";
		// store in list
		for (int i = 0; i < s.length(); i++) {
			ch.add(s.charAt(i) + "");
		}

		// System.out.println(ch);

		for (int i = 0; i < ch.size(); i++) {
			// if there's a space a the front
			if (ch.get(0).equals(" ")) {
				while (ch.get(0).equals(" ")) {
					ch.remove(0);

				}
			}
			// if there's a space at the back
			if (ch.get(ch.size() - 1).equals(" ")) {
				while (ch.get(ch.size() - 1).equals(" ")) {
					ch.remove(ch.size() - 1);
				}
			}

			str = str + ch.get(i);
		}
		return str;
	}

	public static void main(String[] args) {
		String str = "              How are you?   ";
		System.out.println(mytrim(str));
	}
}
